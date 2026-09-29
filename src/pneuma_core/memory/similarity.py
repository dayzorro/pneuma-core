"""用于记忆 embedding 的余弦相似度工具。"""

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量之间的余弦相似度。

    Returns:
        -1.0〜1.0 的相似度。若为零向量则返回 0.0。

    Raises:
        ValueError: 两个向量的维度不一致时。
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions must match: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
