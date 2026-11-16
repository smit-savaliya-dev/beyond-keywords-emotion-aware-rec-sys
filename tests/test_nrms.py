"""Tests for src/nrms.py (sampling inside one impression, batch indexing, padding masks, emotion terms), CPU only with tiny fakes."""
import numpy as np
import torch

from src.nrms import NRMS, gather_batch, sample_groups


def test_negatives_come_from_the_same_impression():
    rng = np.random.default_rng(0)
    cands, labels = np.array([10, 11, 12, 13]), np.array([1, 0, 0, 1])
    groups = sample_groups(cands, labels, 4, rng)
    assert len(groups) == 2 and [g[0] for g in groups] == [10, 13]
    assert all(set(g[1:]) <= {11, 12} for g in groups)
    assert sample_groups(cands, np.ones(4, int), 4, rng) == [] and sample_groups(cands, np.zeros(4, int), 4, rng) == []


def test_gather_batch_indexes_unique_news():
    rng = np.random.default_rng(0)
    cands = [np.array([5, 6, 7]), np.array([8, 9])]
    labels = [np.array([1, 0, 0]), np.array([0, 1])]
    hist = [np.array([1, 2]), np.array([], dtype=np.int64)]
    ids, hidx, hmask, cidx, owner = gather_batch(np.array([0, 1]), hist, cands, labels, 2, rng)
    assert list(owner) == [0, 1] and ids[cidx[0, 0]] == 5 and ids[cidx[1, 0]] == 9
    assert list(ids[hidx[0][hmask[0]]]) == [1, 2] and not hmask[1].any()


def tiny(mode):
    torch.manual_seed(0)
    return NRMS(np.random.default_rng(0).normal(size=(20, 16)).astype(np.float32), mode, emo_dim=3).eval()


def test_padding_does_not_change_the_news_vector():
    m = tiny("none")
    a = m.encode_news(torch.tensor([[3, 4, 5, 0, 0]]))
    b = m.encode_news(torch.tensor([[3, 4, 5, 0, 0, 0, 0]]))
    assert torch.allclose(a, b, atol=1e-5)


def test_user_without_history_scores_zero_and_shapes_hold():
    m = tiny("none")
    v = m.encode_news(torch.tensor([[3, 4, 0], [5, 6, 7]]))
    hist = torch.stack([v, v])
    u = m.encode_user(hist, torch.tensor([[True, True], [False, False]]))
    assert u.shape == (2, 256) and torch.all(u[1] == 0)
    s = m.score(u, torch.stack([v, v]))
    assert s.shape == (2, 2) and torch.all(s[1] == 0)


def test_emotion_terms_start_neutral_and_enter_the_score():
    a, b = tiny("a"), tiny("b")
    tok = torch.tensor([[3, 4, 0]])
    assert not torch.allclose(a.encode_news(tok, torch.ones(1, 3)), a.encode_news(tok, torch.zeros(1, 3)))
    u, c = torch.ones(1, 256), torch.ones(1, 2, 256)
    cong, feats = torch.tensor([[0.5, -0.5]]), torch.ones(1, 3)
    assert torch.allclose(b.score(u, c, cong, feats), b.score(u, c))       # w starts at 0
    with torch.no_grad():
        b.w.fill_(1.0)
    assert not torch.allclose(b.score(u, c, cong, feats), b.score(u, c))


def test_v8_modes_add_a_scaled_personal_match_term():
    """Modes e/f (plan v8): score = dot + (w0 + w1 * consistency) * input; w0 starts at 1, so the input enters at full scale."""
    import numpy as np
    import torch
    from src.nrms import NRMS
    emb = np.random.default_rng(0).random((20, 8)).astype("float32")
    u, c = torch.randn(2, 256), torch.randn(2, 4, 256)
    pm, feats = torch.randn(2, 4), torch.rand(2, 3)
    base = NRMS(emb, "none").score(u, c)
    for mode, n_emo in (("e", 2), ("f", 3)):
        net = NRMS(emb, mode)
        assert len(net.emotion_parameters()) == n_emo
        assert torch.allclose(net.score(u, c, pm, feats), base + pm)
    assert NRMS(emb, "none").emotion_parameters() == []
