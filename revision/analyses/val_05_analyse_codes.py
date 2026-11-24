"""Revision R2 #18: classifier performance against human codes, from the returned coding workbooks.

The returned coding was split: each coder coded part of the 291 items, and items the coders judged
not codable were left blank and are excluded. One item (C150) was coded by both coders; it is used
to report the overlap, and the code of the coder assigned that range is used in the analysis.

Input : data/Jheeng/revision/clinical_repro/coding/coding_key.csv
        data/Jheeng/revision/clinical_repro/coding/returned_final/coding_<coder>_final.xlsx
Output: data/Jheeng/revision/clinical_repro/coding/results/
          merged_codes.csv                one row per coded item: key fields + human codes + coder
          translation_adequacy.csv        voice notes: share rated 3 / 2 / 1, overall and by group
          lexicon_performance.csv         peer support, workaround (primary); urgency, hedging, topics
          sentiment_model.csv             model vs human sentiment
          coder_rates.csv                 code rates by coder (the two halves are random splits)

Precision of the peer-support and workaround lexicons uses every flagged message (sample + census,
unweighted, exact over the corpus). Recall and all other lexicon measures use the 260 sample items
with design weights (1 / inclusion probability); 95% intervals by bootstrap over sample items
(stratified by stratum, 2,000 resamples) or Wilson intervals for unweighted proportions.

Usage: venv/bin/python revision/analyses/val_05_analyse_codes.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.stats import fisher_exact, spearmanr

ROOT = Path(__file__).resolve().parents[2]
CD = ROOT / "data" / "Jheeng" / "revision" / "clinical_repro" / "coding"
RET = CD / "returned_final"
OUT = CD / "results"
CODERS = {"Abdullah": "coding_Abdullah_Bin_Abbas_final.xlsx", "Jawaria": "coding_Jawaria_Irfan_final.xlsx"}
CODES = ["sentiment", "urgency", "peer_support", "workaround", "hedging", "translation_adequacy", "language",
         "topic_training", "topic_coordination", "topic_clinical_protocol", "topic_equipment", "topic_supervision"]
LEX = {"peer_support": "flag_peer_support", "workaround": "flag_workaround", "urgency": "flag_urgency",
       "hedging": "flag_hedging"} | {f"topic_{t}": f"flag_toc_{t}" for t in
                                    ["training", "coordination", "clinical_protocol", "equipment", "supervision"]}
RNG = np.random.default_rng(20261117)
B = 2000


def read_codes(path: Path) -> pd.DataFrame:
    ws = load_workbook(path, data_only=True)["Coding"]
    rows = list(ws.iter_rows(values_only=True))
    df = pd.DataFrame(rows[1:], columns=rows[0]).dropna(how="all").set_index("item_id")
    return df[CODES + ["notes"]]


def norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CODES:
        df[c] = df[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else
                          str(int(v)) if isinstance(v, float) else str(v).strip().lower())
    return df


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, c - h, c + h)


def boot(sample: pd.DataFrame, stat) -> tuple[float, float, float]:
    est = stat(sample)
    groups = [g for _, g in sample.groupby("stratum")]
    vals = []
    for _ in range(B):
        res = pd.concat([g.sample(len(g), replace=True, random_state=int(RNG.integers(1e9))) for g in groups])
        v = stat(res)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (est, np.nan, np.nan)
    return (est, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def wmean(d, mask_num, mask_den):
    w = d["design_weight"]
    den = w[mask_den(d)].sum()
    return np.nan if den == 0 else w[mask_num(d) & mask_den(d)].sum() / den


def main() -> None:
    OUT.mkdir(exist_ok=True)
    key = pd.read_csv(CD / "coding_key.csv").set_index("item_id")
    key = key[key.index.str.startswith("C")]
    coded = {c: norm(read_codes(RET / f)) for c, f in CODERS.items()}
    has = {c: d[[x for x in CODES if x != "translation_adequacy"]].notna().any(axis=1) for c, d in coded.items()}

    # overlap
    both = [i for i in key.index if has["Abdullah"][i] and has["Jawaria"][i]]
    overlap = []
    for i in both:
        for c in CODES:
            overlap.append((i, c, coded["Abdullah"].loc[i, c], coded["Jawaria"].loc[i, c]))
    pd.DataFrame(overlap, columns=["item_id", "code", "Abdullah", "Jawaria"]).to_csv(OUT / "overlap_items.csv", index=False)

    # assignment: Abdullah's range is the first block he coded; Jawaria the rest
    last_a = max(i for i in key.index if has["Abdullah"][i])
    rows = []
    for i in key.index:
        who = "Abdullah" if (has["Abdullah"][i] and i <= last_a) else ("Jawaria" if has["Jawaria"][i] else None)
        if who is None:
            continue
        r = coded[who].loc[i, CODES].to_dict()
        r["item_id"], r["coder"] = i, who
        rows.append(r)
    h = pd.DataFrame(rows).set_index("item_id")
    m = key.join(h, how="inner")
    m["skipped"] = False
    skipped = key.index.difference(m.index)
    m.to_csv(OUT / "merged_codes.csv")
    print(f"coded items: {len(m)} of {len(key)}; skipped as not codable: {len(skipped)}; overlap items: {both}")

    # language group (human code); voice notes form their own group
    def lang_group(r):
        if r.msg_type == "voice_note":
            return "voice note"
        return {"english": "English text", "roman_urdu": "Roman Urdu text", "urdu_script": "Urdu-script text"}.get(r.language, "other/mixed text")
    m["lang_group"] = m.apply(lang_group, axis=1)
    s = m[m.source == "sample"].copy()

    # 1. translation adequacy
    vn = m[m.msg_type == "voice_note"].copy()
    vn = vn[vn.translation_adequacy.isin(["1", "2", "3"])]
    ta = []
    for grp, d in [("all voice notes", vn)] + [(f"phase {p}", d) for p, d in vn.groupby("phase")] + \
                  [(f"coder {c}", d) for c, d in vn.groupby("coder")] + [(f"spoken {l}", d) for l, d in vn.groupby("language")]:
        n = len(d)
        for lev in ["3", "2", "1"]:
            p, lo, hi = wilson((d.translation_adequacy == lev).sum(), n)
            ta.append(dict(group=grp, n=n, level=lev, share=p, lo=lo, hi=hi))
    sv = s[(s.msg_type == "voice_note") & s.translation_adequacy.isin(["1", "2", "3"])]
    for lev in ["3", "2", "1"]:
        est = boot(sv, lambda d, lev=lev: wmean(d, lambda x: x.translation_adequacy == lev, lambda x: x.index == x.index))
        ta.append(dict(group="sample voice notes, design-weighted", n=len(sv), level=lev, share=est[0], lo=est[1], hi=est[2]))
    pd.DataFrame(ta).to_csv(OUT / "translation_adequacy.csv", index=False)

    # 2. lexicons
    lex = []
    for code, flag in LEX.items():
        d = m[m[code].isin(["y", "n"])]
        allflag = d[d[flag] == 1]
        if code in ("peer_support", "workaround"):
            p, lo, hi = wilson((allflag[code] == "y").sum(), len(allflag))
            lex.append(dict(lexicon=code, measure="precision (all flagged, census)", n=len(allflag), est=p, lo=lo, hi=hi))
        ds = s[s[code].isin(["y", "n"])].copy()
        yes = lambda x, c=code: x[c] == "y"
        fl = lambda x, f=flag: x[f] == 1
        measures = {
            "precision (sample, weighted)": lambda x: wmean(x, yes, fl),
            "recall (sample, weighted)": lambda x: wmean(x, fl, yes),
            "specificity (sample, weighted)": lambda x: wmean(x, lambda z: ~fl(z), lambda z: ~yes(z)),
            "human prevalence (sample, weighted)": lambda x: wmean(x, yes, lambda z: z.index == z.index),
            "lexicon prevalence (sample, weighted)": lambda x: wmean(x, fl, lambda z: z.index == z.index),
        }
        if code in ("peer_support", "workaround"):
            # recall using exact flagged true positives (census) and weighted false negatives from unflagged sample items
            tp = int((allflag[code] == "y").sum())
            def rec(x, c=code, f=flag, tp=tp):
                fn = x.loc[(x[f] == 0) & (x[c] == "y"), "design_weight"].sum()
                return np.nan if tp + fn == 0 else tp / (tp + fn)
            measures["recall (census TP + weighted FN)"] = rec
        for name, fn in measures.items():
            est = boot(ds, fn)
            lex.append(dict(lexicon=code, measure=name, n=len(ds), est=est[0], lo=est[1], hi=est[2]))
        lex.append(dict(lexicon=code, measure="n human y in coded set", n=len(d), est=int((d[code] == "y").sum()), lo=np.nan, hi=np.nan))
        lex.append(dict(lexicon=code, measure="n lexicon flags in coded set", n=len(d), est=int((d[flag] == 1).sum()), lo=np.nan, hi=np.nan))
    pd.DataFrame(lex).to_csv(OUT / "lexicon_performance.csv", index=False)

    # 3. sentiment model
    st = []
    d = m[m.sentiment.isin(["positive", "neutral", "negative"]) & m.model_label.notna()].copy()
    d["hp"] = d.sentiment.map({"negative": -1, "neutral": 0, "positive": 1})
    for grp, x in [("all coded", d)] + [(f"type {t}", x) for t, x in d.groupby("msg_type")] + \
                  [(f"language {l}", x) for l, x in d.groupby("lang_group")] + [(f"phase {p}", x) for p, x in d.groupby("phase")]:
        pol = x[x.hp != 0]
        acc = wilson((pol.model_label.str.lower() == pol.sentiment).sum(), len(pol))
        neu = wilson((x.hp == 0).sum(), len(x))
        rho = spearmanr(x.model_score, x.hp).statistic if x.hp.nunique() > 1 and len(x) > 3 else np.nan
        neg_model = wilson((x.model_label == "NEGATIVE").sum(), len(x))
        neg_human = wilson((x.hp == -1).sum(), len(x))
        st.append(dict(group=grp, n=len(x), n_polar=len(pol), accuracy_on_polar=acc[0], acc_lo=acc[1], acc_hi=acc[2],
                       human_neutral_share=neu[0], model_negative_share=neg_model[0], human_negative_share=neg_human[0],
                       spearman_score_vs_human=rho))
    pd.DataFrame(st).to_csv(OUT / "sentiment_model.csv", index=False)

    # 4. coder rates (random halves)
    cr = []
    for c in CODES:
        if c in ("translation_adequacy", "language"):
            continue
        pos = "positive" if c == "sentiment" else "y"
        a = m[m.coder == "Abdullah"]; j = m[m.coder == "Jawaria"]
        ka, na = int((a[c] == pos).sum()), int(a[c].notna().sum())
        kj, nj = int((j[c] == pos).sum()), int(j[c].notna().sum())
        p = fisher_exact([[ka, na - ka], [kj, nj - kj]]).pvalue
        cr.append(dict(code=f"{c} = {pos}", Abdullah=f"{ka}/{na}", Jawaria=f"{kj}/{nj}", fisher_p=p))
    pd.DataFrame(cr).to_csv(OUT / "coder_rates.csv", index=False)
    print("written to", OUT)


if __name__ == "__main__":
    main()
