"""Data preparation, training (checkpointed, resumable, AMP) and scoring for NRMS M6 and M7 (plan v6).

Inputs: news titles (tokenised with the cached all-mpnet-base-v2 tokenizer; its input embeddings initialise the word embeddings, D80),
the GoEmotions raw scores and VAD values, and one part's candidates and histories from src.behaviour_runs.
Outputs: per-epoch checkpoints and logs in data/processed/nrms_runs/<run>/, and one score per candidate row in harness order.
"""
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .behaviour import history_consistency, window_profiles
from .behaviour_runs import load_context
from .config import PROCESSED, SBERT_MODEL, SBERT_REVISION
from .emotion import GOEMOTIONS_LABELS
from .nrms import NRMS, gather_batch
from .subgroups import fast_metrics

RUN_DIR = PROCESSED / "nrms_runs"
TITLE_LEN, MAX_HIST, NEG_K = 30, 50, 4
VAD = ["go_vad_valence", "go_vad_arousal", "go_vad_dominance"]


def news_tables(processed=PROCESSED, cache=True):
    """Token ids (n, 30; 0 = padding, ids shifted by 1), the embedding matrix with a zero padding row, the 31-d emotion inputs
    (28 GoEmotions + 3 VAD, not yet standardised) and the 8-d go_pct vectors, all in news.parquet order."""
    path = processed / "nrms_news_tables.npz"
    if cache and path.exists():
        z = np.load(path)
        return z["tokens"], z["emb"], z["emo"], z["pct"]
    from transformers import AutoModel, AutoTokenizer
    from .config import GO_PCT_COLUMNS
    news = pd.read_parquet(processed / "news.parquet", columns=["news_id", "title"])
    tok = AutoTokenizer.from_pretrained(SBERT_MODEL, revision=SBERT_REVISION)
    enc = tok(news["title"].fillna("").str.lower().tolist(), add_special_tokens=False, truncation=True, max_length=TITLE_LEN)["input_ids"]
    tokens = np.zeros((len(news), TITLE_LEN), dtype=np.int32)
    for i, ids in enumerate(enc):
        tokens[i, :len(ids)] = np.asarray(ids) + 1
    w = AutoModel.from_pretrained(SBERT_MODEL, revision=SBERT_REVISION).get_input_embeddings().weight.detach().numpy()
    emb = np.vstack([np.zeros((1, w.shape[1]), dtype=np.float32), w.astype(np.float32)])
    raw = pd.read_parquet(processed / "news_emotion_raw28_v2.parquet", columns=["news_id", *GOEMOTIONS_LABELS])
    feats = pd.read_parquet(processed / "news_emotion_features_v2.parquet", columns=["news_id", *VAD, *GO_PCT_COLUMNS])
    if raw["news_id"].tolist() != news["news_id"].tolist() or feats["news_id"].tolist() != news["news_id"].tolist():
        raise ValueError("emotion tables are not aligned with news.parquet")
    emo = np.hstack([raw[GOEMOTIONS_LABELS].to_numpy(np.float32), feats[VAD].to_numpy(np.float32)])
    pct = feats[GO_PCT_COLUMNS].to_numpy(np.float32)
    np.savez(path, tokens=tokens, emb=emb, emo=emo, pct=pct)
    return tokens, emb, emo, pct


@dataclass
class PartData:
    """One part (or a day subset of it) as per-impression arrays, plus what scoring needs in harness order."""
    ctx: object
    cands: list          # per impression: item rows
    labels: list         # per impression: labels
    hist: list           # per impression: known history item rows, oldest -> newest
    user_feats: np.ndarray   # per impression: consistency, has-consistency, log(1 + known history)
    profiles: np.ndarray     # per impression: 8-d emotion profile (last 50, decay 0.9)
    days: np.ndarray         # per impression: calendar day


def part_data(part, processed=PROCESSED, perm=None):
    """perm: item emotion permutation (M7ctrl); the profile and consistency are then built from the permuted vectors too."""
    ctx = load_context(part, processed, with_m2=False)
    vec = ctx.vectors if perm is None else ctx.vectors[perm]
    codes, item, y = ctx.imp_codes, ctx.item_idx, ctx.truth["label"].to_numpy()
    cut = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])[1:]
    cands, labels = np.split(item, cut), np.split(y, cut)
    rows = ctx.imps["prof_row"].to_numpy()
    known = [np.asarray(h)[np.asarray(h) >= 0] for h in ctx.hist_rows]
    hist = [known[r][-MAX_HIST:] for r in rows]
    pair, _ = history_consistency(ctx.hist_rows, vec)
    c = pair[rows]
    feats = np.c_[np.nan_to_num(c), np.isfinite(c), np.log1p([len(h) for h in hist])].astype(np.float32)
    prof, _ = window_profiles(ctx.hist_rows, vec, 50, 0.9)
    days = ctx.imps["time"].dt.normalize().to_numpy()
    return PartData(ctx, cands, labels, hist, feats, prof[rows].astype(np.float32), days)


class Tables:
    """News tables on the GPU, with emotion inputs standardised on fit items and an optional fixed item permutation (M7ctrl)."""

    def __init__(self, tokens, emo, pct, fit_items, device, perm=None):
        mu, sd = emo[fit_items].mean(axis=0), emo[fit_items].std(axis=0) + 1e-6
        e = (emo - mu) / sd
        if perm is not None:
            e, pct = e[perm], pct[perm]
        self.tokens = torch.as_tensor(tokens.astype(np.int64), device=device)
        self.emo = torch.as_tensor(e.astype(np.float32), device=device)
        self.pct = torch.as_tensor(pct.astype(np.float32), device=device)
        self.unit_pct = torch.nn.functional.normalize(self.pct, dim=1)


def _cong(tables, profiles, cand_ids):
    """cos(profile, item go_pct) per candidate: profiles (b, 8), cand_ids (b, c)."""
    p = torch.nn.functional.normalize(profiles, dim=1)
    return (tables.unit_pct[cand_ids] * p[:, None, :]).sum(-1)


def train_epoch(model, opt, scaler, data, imps, tables, rng, batch, device, limit=None):
    """One pass over the impressions `imps` in a seeded random order. -> (mean loss, groups seen)."""
    model.train()
    order = rng.permutation(imps)
    if limit:
        order = order[:limit]
    total, n = 0.0, 0
    for s in range(0, len(order), batch):
        b = gather_batch(order[s:s + batch], data.hist, data.cands, data.labels, NEG_K, rng, MAX_HIST)
        if b is None:
            continue
        ids, hidx, hmask, cidx, owner = b
        ids_t = torch.as_tensor(ids, device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            vec = model.encode_news(tables.tokens[ids_t], tables.emo[ids_t])
            u = model.encode_user(vec[torch.as_tensor(hidx, device=device)], torch.as_tensor(hmask, device=device))
            cand = vec[torch.as_tensor(cidx, device=device)]
            cong = feats = None
            if model.mode in ("b", "c"):
                cong = _cong(tables, torch.as_tensor(data.profiles[owner], device=device), ids_t[torch.as_tensor(cidx, device=device)])
                feats = torch.as_tensor(data.user_feats[owner], device=device)
            logits = model.score(u, cand, cong, feats)
            loss = torch.nn.functional.cross_entropy(logits.float(), torch.zeros(len(owner), dtype=torch.long, device=device))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        total += float(loss) * len(owner)
        n += len(owner)
    return total / max(n, 1), n


@torch.no_grad()
def score_part(model, data, tables, device, imps=None, chunk=2048):
    """Score every candidate row (harness order) of the impressions `imps` (default all). Returns a float array of len(truth)."""
    model.eval()
    vecs = []
    for s in range(0, len(tables.tokens), chunk):
        with torch.autocast("cuda", dtype=torch.float16):
            vecs.append(model.encode_news(tables.tokens[s:s + chunk], tables.emo[s:s + chunk]).float())
    vec = torch.cat(vecs)
    imps = np.arange(len(data.cands)) if imps is None else np.asarray(imps)
    out = [None] * len(data.cands)
    for s in range(0, len(imps), 512):
        rows = imps[s:s + 512]
        H = max(1, max(len(data.hist[r]) for r in rows))
        hidx = np.zeros((len(rows), H), dtype=np.int64)
        hmask = np.zeros((len(rows), H), dtype=bool)
        for i, r in enumerate(rows):
            h = data.hist[r]
            hidx[i, :len(h)], hmask[i, :len(h)] = h, True
        with torch.autocast("cuda", dtype=torch.float16):
            u = model.encode_user(vec[torch.as_tensor(hidx, device=device)], torch.as_tensor(hmask, device=device)).float()
        for i, r in enumerate(rows):
            c = torch.as_tensor(data.cands[r], device=device)
            cong = feats = None
            if model.mode in ("b", "c"):
                cong = _cong(tables, torch.as_tensor(data.profiles[r:r + 1], device=device), c[None, :])
                feats = torch.as_tensor(data.user_feats[r:r + 1], device=device)
            out[r] = model.score(u[i:i + 1], vec[c][None], cong, feats)[0].float().cpu().numpy()
    full = np.full(sum(len(c) for c in data.cands), np.nan)
    starts = np.r_[0, np.cumsum([len(c) for c in data.cands])[:-1]]
    for r in imps:
        full[starts[r]:starts[r] + len(data.cands[r])] = out[r]
    return full


def fold_auc(data, scores, imps):
    """Mean per-impression AUC over the impressions `imps` (fast path, identical AUC to the harness)."""
    starts = np.r_[0, np.cumsum([len(c) for c in data.cands])[:-1]]
    rows = np.concatenate([np.arange(starts[r], starts[r] + len(data.cands[r])) for r in imps])
    t = data.ctx.truth.iloc[rows]
    return float(np.nanmean(fast_metrics(t["imp_key"].to_numpy(), t["label"].to_numpy(), scores[rows])["auc"]))


def train_run(name, mode, seed, train, train_imps, valid, valid_imps, tables, emb, device, max_epochs=8, patience=1, batch=32, lr=1e-4,
              limit=None, log=print):
    """Train with early stopping on the valid impressions' AUC; one checkpoint per epoch, resumable. -> (best model, epoch log)."""
    run = RUN_DIR / name
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = NRMS(emb, mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    hist_path = run / "log.json"
    history = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else []
    if history:                                              # resume after the last finished epoch
        state = torch.load(run / f"epoch{len(history)}.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        rng.bit_generator.state = state["rng"]
    best = max(history, key=lambda h: h["valid_auc"]) if history else None
    while len(history) < max_epochs:
        if best and len(history) - best["epoch"] >= patience:
            break
        t0 = time.time()
        loss, n = train_epoch(model, opt, scaler, train, train_imps, tables, rng, batch, device, limit)
        auc = fold_auc(valid, score_part(model, valid, tables, device, valid_imps), valid_imps)
        e = len(history) + 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "rng": rng.bit_generator.state},
                   run / f"epoch{e}.pt")
        history.append({"epoch": e, "train_loss": loss, "groups": n, "valid_auc": auc, "seconds": time.time() - t0})
        hist_path.write_text(json.dumps(history, indent=1), encoding="utf-8")
        log(f"{name} epoch {e}: loss {loss:.4f}, valid AUC {auc:.5f}, {time.time() - t0:.0f}s")
        if best is None or auc > best["valid_auc"]:
            best = history[-1]
    state = torch.load(run / f"epoch{best['epoch']}.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return model, history
