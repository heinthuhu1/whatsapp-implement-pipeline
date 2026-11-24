"""Revision R2 #18: stratified validation sample and coding sheets for bilingual coders.

Input : data/Jheeng/revision/clinical_repro/message_level.csv (from val_01_language_sentiment.py)
Output: data/Jheeng/revision/clinical_repro/
          validation_sample.csv           full sheet: text, translation, model outputs, lexicon flags,
                                          sampling weights, and blank coder columns (analyst copy)
          validation_coding_sheet_blind.csv  same items and blank coder columns, WITHOUT model
                                          outputs or flags (give this one to coders)
          validation_sample_design.csv    stratum sizes, allocation, and inclusion probabilities

Design
  Universe: every non-system message with analysable language content, i.e. free text,
  case-report text, and voice notes with a linked transcript (media placeholders, deleted-message
  notices and unparsed system notices excluded).
  Strata: message type (free text / case-report text / voice note) x language group x phase.
  Language groups: english, roman_urdu, other (urdu_script, mixed, undetermined, none) for text;
  voice notes form one language group because transcription was forced to Urdu.
  Allocation (target 260): voice notes 130, free text 100, case-report text 30, split across
  strata in proportion to stratum size with a floor of min(stratum size, 4).
  Within each stratum, up to half of the allocation is drawn from lexicon-flagged items
  (any of urgency, hedging, peer support, workaround) to oversample rare positives.
  Every item carries its inclusion probability and design weight (1 / probability) so that
  performance estimates can be re-weighted to the corpus.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "Jheeng" / "revision" / "clinical_repro"
SEED = 20261111
TYPE_ALLOC = {"voice_note": 130, "free_text": 100, "case_report_text": 30}
FLAGS = ["flag_urgency", "flag_hedging", "flag_peer_support", "flag_workaround"]
CODER_FIELDS = ["sentiment", "urgency", "peer_support", "workaround", "hedging",
                "translation_adequacy", "language", "notes"]


def lang_group(r) -> str:
    if r["msg_type"] == "voice_note":
        return "voice_ur_transcript"
    return r["language"] if r["language"] in ("english", "roman_urdu") else "other"


def main() -> None:
    rng = np.random.default_rng(SEED)
    d = pd.read_csv(OUT / "message_level.csv")
    d = d[d["msg_type"].isin(TYPE_ALLOC)]
    d = d[~((d["msg_type"] == "voice_note") & d["translation_en"].isna())].copy()
    d["phase"] = d["phase"].astype(str)
    d["lang_group"] = d.apply(lang_group, axis=1)
    d["any_flag"] = d[FLAGS].any(axis=1)
    d["stratum"] = d["msg_type"] + "|" + d["lang_group"] + "|" + d["phase"]

    design, picks = [], []
    for mtype, total in TYPE_ALLOC.items():
        sub = d[d["msg_type"] == mtype]
        sizes = sub.groupby("stratum").size()
        floor = sizes.clip(upper=4)
        remaining = total - floor.sum()
        extra = np.floor((sizes - floor) / (sizes - floor).sum() * max(remaining, 0)).astype(int)
        alloc = (floor + extra).clip(upper=sizes)
        # hand out any rounding remainder to the largest strata
        short = total - alloc.sum()
        for s in sizes.sort_values(ascending=False).index:
            if short <= 0:
                break
            if alloc[s] < sizes[s]:
                alloc[s] += 1
                short -= 1
        for s, k in alloc.items():
            pool = sub[sub["stratum"] == s]
            flagged, unflagged = pool[pool["any_flag"]], pool[~pool["any_flag"]]
            k_f = min(len(flagged), k // 2)
            k_u = min(len(unflagged), k - k_f)
            k_f = min(len(flagged), k - k_u)
            idx_f = rng.choice(flagged.index, size=k_f, replace=False) if k_f else []
            idx_u = rng.choice(unflagged.index, size=k_u, replace=False) if k_u else []
            p_f = k_f / len(flagged) if len(flagged) else np.nan
            p_u = k_u / len(unflagged) if len(unflagged) else np.nan
            for i in idx_f:
                picks.append((i, p_f, "flagged"))
            for i in idx_u:
                picks.append((i, p_u, "unflagged"))
            design.append({"stratum": s, "msg_type": mtype, "N_stratum": len(pool),
                           "N_flagged": len(flagged), "n_sampled": k_f + k_u,
                           "n_sampled_flagged": k_f, "n_sampled_unflagged": k_u,
                           "p_incl_flagged": p_f, "p_incl_unflagged": p_u})
    pd.DataFrame(design).to_csv(OUT / "validation_sample_design.csv", index=False)

    s = d.loc[[i for i, _, _ in picks]].copy()
    s["inclusion_prob"] = [p for _, p, _ in picks]
    s["design_weight"] = 1 / s["inclusion_prob"]
    s["substratum"] = [t for _, _, t in picks]
    s = s.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle presentation order
    s.insert(0, "item_id", [f"V{i + 1:03d}" for i in range(len(s))])
    s["original_text"] = np.where(s["msg_type"] == "voice_note", s["transcription_ur"], s["message"])
    s["english_translation"] = np.where(s["msg_type"] == "voice_note", s["translation_en"], "")
    s["model_label"] = s["label_linked"]
    s["model_prob"] = s["prob_linked"]
    s["model_score"] = s["score_linked"]
    s["model_score_asrun"] = s["score_asrun"]

    meta = ["item_id", "msg_id", "timestamp", "phase", "msg_type", "language", "lang_group", "stratum",
            "substratum", "inclusion_prob", "design_weight", "vn_filename"]
    model = ["model_label", "model_prob", "model_score", "model_score_asrun"] + FLAGS + \
            [c for c in d.columns if c.startswith("flag_toc_") and not c.endswith("_asrun")]
    coder = [f"coder{c}_{f}" for c in (1, 2) for f in CODER_FIELDS]
    adjud = [f"adjudicated_{f}" for f in CODER_FIELDS[:6]]
    full = s[meta + ["original_text", "english_translation"] + model].copy()
    for c in coder + adjud:
        full[c] = ""
    full.to_csv(OUT / "validation_sample.csv", index=False)
    blind = full[["item_id", "phase", "msg_type", "original_text", "english_translation"] + coder]
    blind.to_csv(OUT / "validation_coding_sheet_blind.csv", index=False)

    print(f"[val_02] sample n={len(full)}")
    print(full.groupby(["msg_type", "lang_group", "phase"]).size().to_string())
    print("flag prevalence in sample:", {f: int(full[f].sum()) for f in FLAGS})


if __name__ == "__main__":
    main()
