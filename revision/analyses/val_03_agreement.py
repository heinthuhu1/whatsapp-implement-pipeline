"""Revision R2 #18: inter-coder agreement and classifier performance, once coding is complete.

Usage
  venv/bin/python revision/analyses/val_03_agreement.py            # real codes in validation_sample.csv
  venv/bin/python revision/analyses/val_03_agreement.py --dummy    # self-test on random dummy codes

Input : data/Jheeng/revision/clinical_repro/validation_sample.csv (coder columns filled; the blind
        sheet can be merged back on item_id first, see CODING_GUIDE.md)
Output: data/Jheeng/revision/clinical_repro/agreement_kappa.csv
        data/Jheeng/revision/clinical_repro/classifier_performance.csv
        (prefixed DUMMY_ in --dummy mode)

Coder codes
  sentiment             negative / neutral / positive
  urgency, peer_support, workaround, hedging   y / n
  translation_adequacy  1 (inadequate) / 2 (partly adequate) / 3 (adequate); voice notes only
  language              english / roman_urdu / urdu_script / mixed / other

Statistics
  Agreement: percentage agreement and Cohen's kappa (unweighted; linear-weighted kappa additionally
  for the ordinal sentiment and translation-adequacy codes), with a bootstrap 95% CI (1,000 resamples).
  Consensus: the two coders' code where they agree; otherwise adjudicated_<field> if filled; otherwise
  the item is excluded from classifier evaluation for that field (count reported).
  Lexicon flags vs consensus: precision, recall, F1, specificity, n, n positives.
  Sentiment model (binary POSITIVE/NEGATIVE) vs consensus: (a) accuracy and macro-F1 on items the
  consensus codes as positive or negative; (b) share of consensus-neutral items, which the binary
  model necessarily forces to a polarity; (c) Spearman correlation of model score with consensus
  polarity (-1/0/+1).
  All performance measures are reported overall and by language group, message type and phase,
  unweighted and design-weighted (weights = 1 / inclusion probability from val_02).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "Jheeng" / "revision" / "clinical_repro"
BINARY = {"urgency": "flag_urgency", "peer_support": "flag_peer_support",
          "workaround": "flag_workaround", "hedging": "flag_hedging"}
SENT_MAP = {"negative": -1, "neutral": 0, "positive": 1}


def cohen_kappa(a, b, weights: str | None = None, labels=None) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return np.nan
    labels = sorted(set(a) | set(b)) if labels is None else labels
    k = len(labels)
    if k < 2:
        return np.nan
    idx = {l: i for i, l in enumerate(labels)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    if weights == "linear":
        W = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    else:
        W = 1 - np.eye(k)
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else np.nan


def boot_ci(a, b, fn, n=1000, seed=1):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 5:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        vals.append(fn(a[i], b[i]))
    vals = np.array([v for v in vals if not np.isnan(v)])
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) else (np.nan, np.nan)


def binary_perf(pred, truth, w=None) -> dict:
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    w = np.ones(len(pred)) if w is None else np.asarray(w, float)
    tp = w[pred & truth].sum(); fp = w[pred & ~truth].sum()
    fn = w[~pred & truth].sum(); tn = w[~pred & ~truth].sum()
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and not np.isnan(prec) and not np.isnan(rec) else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {"n": int(len(pred)), "n_pos_consensus": int(truth.sum()), "n_flagged": int(pred.sum()),
            "precision": prec, "recall": rec, "f1": f1, "specificity": spec}


def norm_code(s: pd.Series, field: str) -> pd.Series:
    s = s.astype(str).str.strip().str.lower().replace({"nan": np.nan, "": np.nan})
    if field in BINARY:
        s = s.replace({"yes": "y", "no": "n", "1": "y", "0": "n", "true": "y", "false": "n"})
    if field == "translation_adequacy":
        s = s.str.replace(r"\.0$", "", regex=True)
    return s


def consensus(df: pd.DataFrame, field: str) -> pd.Series:
    a = norm_code(df[f"coder1_{field}"], field)
    b = norm_code(df[f"coder2_{field}"], field)
    adj = norm_code(df.get(f"adjudicated_{field}", pd.Series(np.nan, index=df.index)), field)
    return a.where(a == b, adj)


def make_dummy(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Random codes with ~80% coder agreement and codes loosely tied to the model outputs."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    base_sent = np.where(d["model_label"] == "POSITIVE", "positive", "negative")
    base_sent = np.where(rng.random(len(d)) < 0.25, "neutral", base_sent)
    for c in (1, 2):
        flip = rng.random(len(d)) < 0.2
        d[f"coder{c}_sentiment"] = np.where(flip, rng.choice(list(SENT_MAP), len(d)), base_sent)
        for f, flag in BINARY.items():
            truth = np.where(rng.random(len(d)) < 0.75, d[flag].astype(bool), rng.random(len(d)) < 0.1)
            flip = rng.random(len(d)) < 0.1
            d[f"coder{c}_{f}"] = np.where(truth ^ flip, "y", "n")
        ta = rng.choice(["1", "2", "3"], len(d), p=[0.1, 0.3, 0.6])
        d[f"coder{c}_translation_adequacy"] = np.where(d["msg_type"] == "voice_note", ta, "")
        d[f"coder{c}_language"] = d["language"]
    return d


def main(dummy: bool = False) -> None:
    df = pd.read_csv(OUT / "validation_sample.csv", dtype=str, keep_default_na=False)
    for c in ("design_weight", "model_score", "model_prob"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for f in BINARY.values():
        df[f] = df[f].astype(str).str.lower().eq("true")
    if dummy:
        df = make_dummy(df)
    prefix = "DUMMY_" if dummy else ""
    filled = df["coder1_sentiment"].astype(str).str.strip().ne("").sum()
    if filled == 0:
        sys.exit("No coder codes found. Fill the coder columns first, or run with --dummy.")

    groupings = [("Overall", None), ("lang_group", "lang_group"), ("msg_type", "msg_type"), ("phase", "phase")]

    # ---------------- inter-coder agreement
    rows = []
    for field in ["sentiment", "translation_adequacy", "language"] + list(BINARY):
        a_all = norm_code(df[f"coder1_{field}"], field)
        b_all = norm_code(df[f"coder2_{field}"], field)
        for gname, gcol in groupings:
            groups = [("all", df.index)] if gcol is None else list(df.groupby(gcol).groups.items())
            for gval, idx in groups:
                a, b = a_all.loc[idx], b_all.loc[idx]
                ok = a.notna() & b.notna()
                a, b = a[ok].values, b[ok].values
                r = {"field": field, "grouping": gname, "group": gval, "n_double_coded": len(a),
                     "pct_agreement": float((a == b).mean() * 100) if len(a) else np.nan,
                     "kappa": cohen_kappa(a, b)}
                r["kappa_ci_low"], r["kappa_ci_high"] = boot_ci(a, b, cohen_kappa)
                if field in ("sentiment", "translation_adequacy"):
                    labels = list(SENT_MAP) if field == "sentiment" else ["1", "2", "3"]
                    fn = lambda x, y, L=labels: cohen_kappa(x, y, "linear", L)
                    r["kappa_linear_weighted"] = fn(a, b)
                rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / f"{prefix}agreement_kappa.csv", index=False)

    # ---------------- classifier performance vs consensus
    rows = []
    for field, flag in BINARY.items():
        cons = consensus(df, field)
        for gname, gcol in groupings:
            groups = [("all", df.index)] if gcol is None else list(df.groupby(gcol).groups.items())
            for gval, idx in groups:
                sub = df.loc[idx]
                c = cons.loc[idx]
                ok = c.notna()
                for weighted in (False, True):
                    r = binary_perf(sub.loc[ok, flag], c[ok] == "y",
                                    sub.loc[ok, "design_weight"] if weighted else None)
                    rows.append({"classifier": f"lexicon_{field}", "grouping": gname, "group": gval,
                                 "weighted": weighted, "n_no_consensus": int((~ok).sum()), **r})
    cons = consensus(df, "sentiment")
    for gname, gcol in groupings:
        groups = [("all", df.index)] if gcol is None else list(df.groupby(gcol).groups.items())
        for gval, idx in groups:
            sub, c = df.loc[idx], cons.loc[idx]
            ok = c.notna()
            polar = ok & c.isin(["negative", "positive"])
            pred_pos = sub.loc[polar, "model_label"].eq("POSITIVE").values
            true_pos = c[polar].eq("positive").values
            for weighted in (False, True):
                w = sub.loc[polar, "design_weight"].values if weighted else np.ones(polar.sum())
                acc = float((w * (pred_pos == true_pos)).sum() / w.sum()) if polar.sum() else np.nan
                p_pos = binary_perf(pred_pos, true_pos, w)
                p_neg = binary_perf(~pred_pos, ~true_pos, w)
                macro = np.nanmean([p_pos["f1"], p_neg["f1"]]) if polar.sum() else np.nan
                pol = c[ok].map(SENT_MAP).astype(float)
                rho = pd.Series(sub.loc[ok, "model_score"].values).corr(pd.Series(pol.values), method="spearman") if ok.sum() > 2 else np.nan
                rows.append({"classifier": "distilbert_sst2", "grouping": gname, "group": gval, "weighted": weighted,
                             "n": int(ok.sum()), "n_polar_consensus": int(polar.sum()),
                             "n_neutral_consensus": int((c[ok] == "neutral").sum()),
                             "pct_neutral_forced_to_polarity": float((c[ok] == "neutral").mean() * 100) if ok.sum() else np.nan,
                             "accuracy_polar": acc, "macro_f1_polar": macro,
                             "precision": p_neg["precision"], "recall": p_neg["recall"], "f1": p_neg["f1"],
                             "spearman_score_vs_polarity": rho, "n_no_consensus": int((~ok).sum()),
                             "note": "precision/recall/F1 are for the NEGATIVE class"})
    perf = pd.DataFrame(rows)
    perf.to_csv(OUT / f"{prefix}classifier_performance.csv", index=False)

    ka = pd.read_csv(OUT / f"{prefix}agreement_kappa.csv")
    print(ka[ka["grouping"] == "Overall"][["field", "n_double_coded", "pct_agreement", "kappa", "kappa_ci_low", "kappa_ci_high"]].round(3).to_string(index=False))
    print(perf[(perf["grouping"] == "Overall") & (~perf["weighted"])][["classifier", "n", "precision", "recall", "f1"]].round(3).to_string(index=False))
    print(f"[val_03] wrote {prefix}agreement_kappa.csv and {prefix}classifier_performance.csv to {OUT}")


if __name__ == "__main__":
    main(dummy="--dummy" in sys.argv)
