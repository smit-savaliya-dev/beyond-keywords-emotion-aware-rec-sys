"""NRMS (Wu et al., 2019) for M6, and its emotion variants for M7 (plan v6, stages 3 and 4).

News encoder: word embeddings -> dropout -> multi-head self-attention (16 heads x 16 dims) -> dropout -> additive attention (query 200) -> 256-d.
User encoder: multi-head self-attention over the clicked-news vectors -> additive attention. Score = dot(user, news).
M7a adds a small linear projection of the item emotion vector to every news vector; M7b adds g(user) * w * cos(user emotion profile, item emotion),
with g a small gate on the user's emotional consistency and history length; M7c does both. M7ctrl uses item emotion vectors permuted across items.

Inputs: token ids of every title, history rows, candidate rows with labels. Outputs: trained weights and scores per candidate.
Training uses one clicked candidate plus K non-clicks sampled from the same impression (softmax cross-entropy over K + 1).
"""
import math

import numpy as np
import torch
from torch import nn

NEWS_DIM = 256


class SelfAttention(nn.Module):
    """Multi-head self-attention with a padding mask; heads x head_dim output."""

    def __init__(self, in_dim, heads=16, head_dim=16):
        super().__init__()
        self.h, self.d = heads, head_dim
        self.q, self.k, self.v = (nn.Linear(in_dim, heads * head_dim) for _ in range(3))

    def forward(self, x, mask):
        b, n, _ = x.shape
        q, k, v = (f(x).view(b, n, self.h, self.d).transpose(1, 2) for f in (self.q, self.k, self.v))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.d)
        att = att.masked_fill(~mask[:, None, None, :], torch.finfo(att.dtype).min)
        out = torch.softmax(att.float(), dim=-1).to(v.dtype) @ v
        return out.transpose(1, 2).reshape(b, n, self.h * self.d)


class AdditiveAttention(nn.Module):
    """Weighted sum of a sequence with weights softmax(q . tanh(W x + b)), padding masked."""

    def __init__(self, in_dim, query_dim=200):
        super().__init__()
        self.proj = nn.Linear(in_dim, query_dim)
        self.query = nn.Parameter(torch.randn(query_dim) * 0.01)

    def forward(self, x, mask):
        a = torch.tanh(self.proj(x)) @ self.query
        a = a.masked_fill(~mask, torch.finfo(a.dtype).min)
        w = torch.softmax(a.float(), dim=-1).to(x.dtype)
        return (w.unsqueeze(-1) * x).sum(dim=1)


class NRMS(nn.Module):
    """NRMS with optional emotion: mode in {'none', 'a', 'b', 'c'}."""

    def __init__(self, embeddings, mode="none", emo_dim=31, dropout=0.2):
        super().__init__()
        self.mode = mode
        self.emb = nn.Embedding.from_pretrained(torch.as_tensor(embeddings, dtype=torch.float32), freeze=False, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.news_att = SelfAttention(self.emb.embedding_dim)
        self.news_pool = AdditiveAttention(NEWS_DIM)
        self.user_att = SelfAttention(NEWS_DIM)
        self.user_pool = AdditiveAttention(NEWS_DIM)
        if mode in ("a", "c"):
            self.emo_proj = nn.Linear(emo_dim, NEWS_DIM, bias=False)
            nn.init.normal_(self.emo_proj.weight, std=0.01)
        if mode in ("b", "c"):
            self.gate = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
            self.w = nn.Parameter(torch.zeros(1))

    def encode_news(self, tokens, emo=None):
        """tokens: (n, L) ids, 0 = padding; emo: (n, emo_dim) or None. -> (n, 256)."""
        mask = tokens > 0
        mask[:, 0] = True                                       # an empty title still has one position
        x = self.drop(self.emb(tokens))
        x = self.drop(self.news_att(x, mask))
        v = self.news_pool(x, mask)
        if self.mode in ("a", "c") and emo is not None:
            v = v + self.emo_proj(emo)
        return v

    def encode_user(self, hist_vecs, hist_mask):
        """hist_vecs: (b, H, 256); hist_mask: (b, H). A user with no history gets a zero vector."""
        has = hist_mask.any(dim=1)
        m = hist_mask.clone()
        m[~has, 0] = True
        x = self.user_att(hist_vecs, m)
        u = self.user_pool(x, m)
        return u * has[:, None].to(u.dtype)

    def score(self, user, cand, cong=None, user_feats=None):
        """user: (b, 256); cand: (b, c, 256); cong: (b, c) cosine of the user emotion profile with each candidate; user_feats: (b, 3)."""
        s = (cand * user[:, None, :]).sum(-1)
        if self.mode in ("b", "c") and cong is not None:
            s = s + torch.sigmoid(self.gate(user_feats)) * self.w * cong
        return s


def sample_groups(cands, labels, k, rng):
    """Training groups from one impression: every click with k non-clicks from the same impression (with replacement when fewer).
    -> list of index arrays (first = the click); empty if the impression has no click or no non-click."""
    pos, neg = cands[labels == 1], cands[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return []
    return [np.r_[p, rng.choice(neg, size=k, replace=len(neg) < k)] for p in pos]


def gather_batch(rows, hist, cands, labels, k, rng, max_hist=50):
    """One training batch from impression rows. -> (unique news ids, history index (b, H) into them, mask, candidate index (b, k + 1), impression row per group)."""
    groups, owner = [], []
    for r in rows:
        for g in sample_groups(cands[r], labels[r], k, rng):
            groups.append(g)
            owner.append(r)
    if not groups:
        return None
    owner = np.asarray(owner)
    hists = [hist[r][-max_hist:] for r in owner]
    ids = np.unique(np.concatenate([np.concatenate(groups), *[h for h in hists if len(h)]]))
    pos = {v: i for i, v in enumerate(ids)}
    H = max(1, max(len(h) for h in hists))
    hidx = np.zeros((len(owner), H), dtype=np.int64)
    hmask = np.zeros((len(owner), H), dtype=bool)
    for i, h in enumerate(hists):
        hidx[i, :len(h)] = [pos[v] for v in h]
        hmask[i, :len(h)] = True
    cidx = np.array([[pos[v] for v in g] for g in groups], dtype=np.int64)
    return ids, hidx, hmask, cidx, owner
