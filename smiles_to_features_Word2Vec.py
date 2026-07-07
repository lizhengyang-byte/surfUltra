"""
基于 Word2Vec 的分子特征化方法。

工作流程：
  1. 从训练数据中读取所有 SMILES，分词后训练 Word2Vec 模型，
     使得每个化学 token（原子、离子、特殊符号等）获得一个稠密向量。
  2. 对于任意输入 SMILES，先分词，再取所有 token 向量的平均值，
     得到一个固定维度的特征向量（embedding_dim 维）。
  3. 该特征向量可作为后续机器学习模型（LightGBM、XGBoost、MLP 等）的输入。

使用方式（在训练脚本中）：
    from smiles_to_features_Word2Vec import Word2VecFeaturizer
    featurizer = Word2VecFeaturizer(smiles_list=df["SMILES"].tolist(),
                                    embedding_dim=100)
    featurizer.train()
    vec = featurizer.transform("CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]")
"""

import re
import numpy as np
from gensim.models import Word2Vec
from typing import List, Optional, Tuple


# ========== SMILES 分词 ==========

# 正则：优先匹配 [原子或离子]（如 [Na+], [O-], [NH4+]），
# 然后匹配双字母原子（Cl, Br, Si, Se 等），
# 再匹配单字母原子（C, N, O, S, P, F, I, B 等），
# 最后匹配其它符号（=, #, (, ), 1, 2, +, -, ., /, \ 等）。
SMILES_TOKEN_REGEX = re.compile(
    r"""
    \[[^\]]+\]      # 括号内的原子或离子，如 [Na+], [O-], [NH4+]
    |Br             # 溴
    |Cl             # 氯
    |Si             # 硅
    |Se             # 硒
    |Te             # 碲
    |As             # 砷
    |Sb             # 锑
    |Bi             # 铋
    |Po             # 钋
    |At             # 砹
    |Rn             # 氡
    |Fr             # 钫
    |Ra             # 镭
    |Ac             # 锕
    |Th             # 钍
    |Pa             # 镤
    |U              # 铀
    |Np             # 镎
    |Pu             # 钚
    |Am             # 镅
    |Cm             # 锔
    |Bk             # 锫
    |Cf             # 锎
    |Es             # 锿
    |Fm             # 镄
    |Md             # 钔
    |No             # 锘
    |Lr             # 铹
    |Rf             # 𬬻
    |Db             # 𬭊
    |Sg             # 𬭳
    |Bh             # 𬭛
    |Hs             # 𬭶
    |Mt             # 鿏
    |Ds             # 𬭬
    |Rg             # 𬬭
    |Cn             # 鎶
    |Nh             # 鉨
    |Fl             # 𫓧
    |Mc             # 镆
    |Lv             # 𫟷
    |Ts             # 鿬
    |Og             # 鿫
    |La             # 镧
    |Ce             # 铈
    |Pr             # 镨
    |Nd             # 钕
    |Pm             # 钷
    |Sm             # 钐
    |Eu             # 铕
    |Gd             # 钆
    |Tb             # 铽
    |Dy             # 镝
    |Ho             # 钬
    |Er             # 铒
    |Tm             # 铥
    |Yb             # 镱
    |Lu             # 镥
    |Hf             # 铪
    |Ta             # 钽
    |W              # 钨
    |Re             # 铼
    |Os             # 锇
    |Ir             # 铱
    |Pt             # 铂
    |Au             # 金
    |Hg             # 汞
    |Tl             # 铊
    |Pb             # 铅
    |[A-Za-z]       # 单个字母（C, N, O, S, P, F, I, H, c, n, o, s 等）
    |\d             # 数字（环编号如 1, 2, 3...）
    |[=#@+\-\\/()\[\]%\.]  # 其它符号
    """,
    re.VERBOSE,
)


def tokenize_smiles(smiles: str) -> List[str]:
    """
    将 SMILES 字符串分词为 token 列表。

    Parameters
    ----------
    smiles : str
        SMILES 字符串。

    Returns
    -------
    List[str] : token 列表。
    """
    return SMILES_TOKEN_REGEX.findall(smiles)


# ========== Word2Vec 特征化器 ==========


class Word2VecFeaturizer:
    """
    使用 Word2Vec 将 SMILES 分子转化为固定维度的特征向量。

    原理：
      - 对每个 SMILES 进行分词，得到 token 序列
      - 用 gensim 的 Word2Vec 在大量 SMILES 上训练词向量
      - 对于新的分子，将其所有 token 的向量平均作为分子特征向量

    Parameters
    ----------
    smiles_list : List[str]
        用于训练 Word2Vec 的 SMILES 列表。
    embedding_dim : int, default=100
        Word2Vec 向量的维度。
    window : int, default=5
        Word2Vec 的上下文窗口大小。
    min_count : int, default=1
        最少出现次数，低于此频率的 token 将被忽略。
    workers : int, default=4
        训练用的线程数。
    sg : int, default=0
        Word2Vec 算法：0=CBOW, 1=Skip-gram。
    """

    def __init__(
        self,
        smiles_list: Optional[List[str]] = None,
        embedding_dim: int = 100,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        sg: int = 0,
    ):
        self.smiles_list = smiles_list or []
        self.embedding_dim = embedding_dim
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.model: Optional[Word2Vec] = None
        self._tokens_cache: List[List[str]] = []

    def tokenize_corpus(self, smiles_list: Optional[List[str]] = None) -> List[List[str]]:
        """
        对 SMILES 列表进行分词，返回 token 序列列表。

        Parameters
        ----------
        smiles_list : List[str], optional
            如果为 None，则使用初始化时传入的 smiles_list。

        Returns
        -------
        List[List[str]] : 分词后的 token 序列列表。
        """
        source = smiles_list if smiles_list is not None else self.smiles_list
        return [tokenize_smiles(smi) for smi in source]

    def train(self, smiles_list: Optional[List[str]] = None) -> "Word2VecFeaturizer":
        """
        在 SMILES 语料库上训练 Word2Vec 模型。

        Parameters
        ----------
        smiles_list : List[str], optional
            用于训练的 SMILES 列表。如果为 None，则使用初始化时传入的。

        Returns
        -------
        self : Word2VecFeaturizer
        """
        if smiles_list is not None:
            self.smiles_list = smiles_list

        if len(self.smiles_list) == 0:
            raise ValueError("SMILES 列表为空，无法训练 Word2Vec。")

        print(f"[Word2Vec] 正在分词 {len(self.smiles_list)} 个 SMILES ...")
        self._tokens_cache = self.tokenize_corpus()

        print(f"[Word2Vec] 正在训练 Word2Vec (dim={self.embedding_dim}, "
              f"window={self.window}, min_count={self.min_count}, sg={self.sg}) ...")
        self.model = Word2Vec(
            sentences=self._tokens_cache,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
        )
        print(f"[Word2Vec] 训练完成！词汇表大小: {len(self.model.wv)}")
        return self

    def transform(self, smiles: str) -> np.ndarray:
        """
        将单个 SMILES 转化为特征向量（所有 token 向量的平均值）。

        Parameters
        ----------
        smiles : str
            SMILES 字符串。

        Returns
        -------
        np.ndarray : 形状为 (embedding_dim,) 的特征向量。
        """
        if self.model is None:
            raise RuntimeError("Word2Vec 模型尚未训练，请先调用 train()。")

        tokens = tokenize_smiles(smiles)
        vectors = []

        for token in tokens:
            if token in self.model.wv:
                vectors.append(self.model.wv[token])

        if len(vectors) == 0:
            # 如果所有 token 都不在词表中，返回零向量
            return np.zeros(self.embedding_dim)

        return np.mean(vectors, axis=0)

    def transform_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        批量转化多个 SMILES 为特征向量。

        Parameters
        ----------
        smiles_list : List[str]
            SMILES 字符串列表。

        Returns
        -------
        np.ndarray : 形状为 (n_smiles, embedding_dim) 的特征矩阵。
        """
        return np.array([self.transform(smi) for smi in smiles_list])

    def get_token_vector(self, token: str) -> np.ndarray:
        """
        获取单个 token 的向量。

        Parameters
        ----------
        token : str
            化学 token（如 'C', 'O', '[Na+]', 'Cl' 等）。

        Returns
        -------
        np.ndarray : 形状为 (embedding_dim,) 的向量。
        """
        if self.model is None:
            raise RuntimeError("Word2Vec 模型尚未训练，请先调用 train()。")
        return self.model.wv[token]

    def get_most_similar(self, token: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        查找与给定 token 最相似的 token。

        Parameters
        ----------
        token : str
            化学 token。
        topn : int, default=10
            返回的最相似 token 数量。

        Returns
        -------
        List[Tuple[str, float]] : (token, 相似度) 列表。
        """
        if self.model is None:
            raise RuntimeError("Word2Vec 模型尚未训练，请先调用 train()。")
        return self.model.wv.most_similar(token, topn=topn)


# ========== 便捷函数 ==========

def smiles_to_word2vec_features(
    smiles_list: List[str],
    embedding_dim: int = 100,
    window: int = 5,
    min_count: int = 1,
) -> Word2VecFeaturizer:
    """
    便捷函数：从一个 SMILES 列表训练 Word2Vec 并返回特征化器。

    Parameters
    ----------
    smiles_list : List[str]
        用于训练的 SMILES 列表。
    embedding_dim : int, default=100
        嵌入向量维度。
    window : int, default=5
        Word2Vec 上下文窗口。
    min_count : int, default=1
        最小 token 出现次数。

    Returns
    -------
    Word2VecFeaturizer : 已训练的特征化器。
    """
    featurizer = Word2VecFeaturizer(
        smiles_list=smiles_list,
        embedding_dim=embedding_dim,
        window=window,
        min_count=min_count,
    )
    featurizer.train()
    return featurizer


# ========== 主函数（演示） ==========

def main():
    """
    演示如何使用 Word2VecFeaturizer。
    """
    # 模拟一些 SMILES（实际应从数据文件读取）
    demo_smiles = [
        "CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]",
        "CC(C)(C)OCCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]",
        "O=S(=O)([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F.[Li+]",
        "CCCCCCCCCCCCS(=O)(=O)[O-].[K+]",
        "CCCCCCCCCCCCS(=O)(=O)[O-].[Li+]",
        "O=C([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F.[Na+]",
        "CCCCCCCCCCCCOCC(O)CO",
        "CCCCCCCCCCCCCCCCCCCC(=O)OCC(O)CO",
    ]

    print("=" * 60)
    print("Word2Vec 分子特征化 演示")
    print("=" * 60)

    # 1. 训练
    print(f"\n训练集大小: {len(demo_smiles)} 个 SMILES")
    print(f"嵌入维度 : {100}")
    featurizer = smiles_to_word2vec_features(
        demo_smiles,
        embedding_dim=100,
        window=5,
        min_count=1,
    )

    # 2. 对一个分子提取特征
    test_smiles = "CC(C)(C)OCCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]"
    print(f"\n测试 SMILES: {test_smiles}")
    print(f"分词结果 : {tokenize_smiles(test_smiles)}")

    feature_vector = featurizer.transform(test_smiles)
    print(f"特征向量维度 : {len(feature_vector)}")
    print(f"特征向量 (前 10 个元素): {feature_vector[:10]}")

    # 3. 批量转化
    batch_features = featurizer.transform_batch(demo_smiles)
    print(f"\n批量特征矩阵形状: {batch_features.shape}")

    # 4. 相似 token 查询（展示 Word2Vec 的语义学习能力）
    print("\n--- Token 相似度示例 ---")
    for test_token in ["C", "O", "[Na+]", "F", "=O"]:
        if test_token in featurizer.model.wv:
            similar = featurizer.get_most_similar(test_token, topn=3)
            print(f"'{test_token}' 最相似的 token: {similar}")
        else:
            print(f"'{test_token}' 不在词表中")

    print("\n演示完成！")


if __name__ == "__main__":
    main()