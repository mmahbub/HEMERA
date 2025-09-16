import pickle
import torch as torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from torch import Tensor
from sparsemax import Sparsemax

def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_sparsemax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    sparsemax = Sparsemax(dim=-1)
    if valid_lens is None:
        return sparsemax(X)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return sparsemax(X.reshape(shape), dim=-1)

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class D2LAdditiveAttention(nn.Module):
    """Additive attention."""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(D2LAdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class D2LDotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout, use_sparsemax, **kwargs):
        super(D2LDotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.use_sparsemax = use_sparsemax

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        if (self.use_sparsemax):
            self.attention_weights = masked_sparsemax(scores, valid_lens)
        else:
            self.attention_weights = masked_softmax(scores, valid_lens)
        self.attention_weights.requires_grad_()
        self.attention_weights.retain_grad()
        return torch.bmm(self.dropout(self.attention_weights), values)

def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads."""
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`."""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class LinformerAttention(nn.Module):
    """Multi-head attention."""
    def __init__(self, embed_dim, seq_len, linform_k,
                 num_heads, dropout, use_sparsemax=False, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = D2LDotProductAttention(dropout, use_sparsemax)

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=bias)
        # Linformer components
        self.E_i = nn.Linear(seq_len, linform_k)
        self.F_i = nn.Linear(seq_len, linform_k)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # Linformer approximation
        keys = self.E_i(keys.swapaxes(1,2)).swapaxes(1,2)
        values = self.F_i(values.swapaxes(1,2)).swapaxes(1,2)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)

        output_concat = transpose_output(output, self.num_heads)
        return self.attention.attention_weights, self.W_o(output_concat)


class D2LMultiHeadAttention(nn.Module):
    """Multi-head attention."""
    def __init__(self, embed_dim,
                 num_heads, dropout, device='cuda', bias=False, **kwargs):
        super(D2LMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = D2LDotProductAttention(dropout)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.attention.attention_weights, self.W_o(output_concat)

class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        print('d_model: ', d_model)
        self.P = torch.zeros((1, int(max_len), d_model))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, d_model, 2, dtype=torch.float32) / d_model)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].requires_grad_(False).to(X.device)
        return self.dropout(X)

class AddNorm(nn.Module):
    """Add residual then normalise"""
    def __init__(self, normalized_shape, device='cuda', dropout=0.0, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape, device=device)
        
    def forward(self, residual, X):
        return self.ln(self.dropout(X) + residual)
    
class PositionWiseFFN(nn.Module):
    def __init__(self, num_in, num_hidden, num_out, device='cuda') -> None:
        super().__init__()
        self.dense1 = nn.Linear(num_in, num_hidden, device=device) 
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(num_in, num_hidden, device=device)
    
    def forward(self, X):
        return self.dense2(self.gelu(self.dense1(X)))