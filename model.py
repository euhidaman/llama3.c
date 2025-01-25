import math
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 64
    n_layers: int = 5
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 512
    multiple_of: int = 4
    norm_eps: float = 1e-5
    max_seq_len: int = 512
    dropout: float = 0.05


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_kv_heads
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads *
                            self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads *
                            self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        # Apply attention mechanism
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / \
            math.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            args.dim, args.dim * 4, args.multiple_of, args.dropout)
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight  # weight tying

    def forward(self, tokens, targets=None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        if targets is not None:
            logits = self.output(h)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.output(h[:, [-1], :])
            return logits
