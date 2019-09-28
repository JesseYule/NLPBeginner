import torch
import torch.nn.functional as F
import math


def attention(query, key, value, mask=None, dropout=0.0):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)

    # Scaled Dot-Product
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 归一化处理
    p_attn = F.softmax(scores, dim=-1)

    # (Dropout described below)
    p_attn = F.dropout(p_attn, p=dropout)

    return torch.matmul(p_attn, value), p_attn
