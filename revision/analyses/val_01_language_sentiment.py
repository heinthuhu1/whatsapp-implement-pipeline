"""Revision R2 #18: message-level language, message type, lexicon flags and sentiment.

Reads ONLY the archived Jheeng data (data/Jheeng/...). Writes ONLY to
data/Jheeng/revision/clinical_repro/. Nothing is sent to any external service:
the DistilBERT model is loaded from the local Hugging Face cache with
HF_HUB_OFFLINE=1.

Outputs
  message_level.csv              one row per non-system message (contains message text; local only)
  lang_composition.csv           corpus composition by language x type x phase
  lang_flags_sentiment.csv       lexicon flag rates and mean sentiment by language x type x phase
  lang_sentiment_by_phase.csv    phase means, as-run vs linked translations, and with/without voice notes
  asrun_replication_check.txt    check that the replicated as-run scores reproduce sentiment_metrics.csv

Two sentiment scores are computed for every message:
  score_asrun   replicates src/05_sentiment.py exactly on the archived inputs. In the archived
                run the voice-note filename join failed (see below), so voice notes were scored
                on their placeholder text, e.g. "PTT-[...]-WA0005.opus (file attached)".
  score_linked  voice notes are scored on their Whisper English translation, linked by
                (PTT/AUD prefix, message date, WA sequence number). Text messages are unchanged.

Why the join failed: 00_anonymise.py redacts runs of 7+ digits as [NUMBER]; one sender-lookup
entry itself contains the token "NUMBER", so the name-redaction pass then rewrote [NUMBER] as
a coded ID. The yyyymmdd date inside every voice-note filename was therefore lost, and the
filename extracted by 01_parse.py ("-WA0005.opus") never matches voice_notes.csv
("PTT-20240924-WA0005.opus").
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
J = ROOT / "data" / "Jheeng"
OUT = J / "revision" / "clinical_repro"
OUT.mkdir(parents=True, exist_ok=True)

PHASES = ["2024", "2025", "2026"]
WINDOW = ("2024-09-22", "2026-04-11")


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- language heuristic
ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
LATIN_RE = re.compile(r"[A-Za-z]")
WORD_RE = re.compile(r"[a-z']+")

# High-precision Roman Urdu function words and frequent lexical items. Tokens that are
# also common English words (e.g. "me", "is", "to", "us", "main", "the", "hi") are excluded.
ROMAN_URDU_MARKERS = set("""
hai hain hy hn ka ki ke ko se mein mai mei nahi nahin nai nhi aur ap aap apna apne apni apko aapko
kar kr karo karen karna krna kiya kia kya kyun kyon kab kahan ho hoga hogi hogya hogaya tha thi
bhi bhai jee ji sb sab yeh ye wo woh hum ham mujhe mjhe muje tum unka uska unki uski abhi phir
wala wali waly gaya gya gayi gai raha rahi rha rhi rahe acha achha accha theek thik shukriya
jazakallah inshallah insha mashallah alhamdulillah alhamdulilah assalam asalam salam walaikum
alaikum aoa haan jaldi aaj aj kal din waqt bohat bahut bht liye lye liay sath saath paas pas koi
kuch kuchh bata batao btao dekh dekho chalo lekin magar agar jab tab mera meri mere hamara hamari
inko unko isko usko wapis wapas gari gaari mareez mariz bukhar dard chot hoa hua hui huwa diya dia
gye gaye lagta lag rahy rhy krain krein karein krty karte karta karti krta
ha hy ny sy ma pa ab ja ra hm mn ga gi gy ky kay bhe lea liya behtr behtar kry kre kren krna
nikl jaye jay gay wah mashalla zabordast zabardast kary khair khar kaha kaise kese kesy kaisa yahan wahan idhar udhar zara thora thoda
""".split())

ENGLISH_MARKERS = set("""
the are was were been being and or of in on at for with from by this that these those it its we you
your our they he she his her have has had will would should can could please thank thanks today
tomorrow yesterday yes not all any what when where who how there here do does did done my their them
which also very just about after before into than then only because but if so get got going
good work job luck ok okay excellent done well great nice news check think know sorry welcome
""".split())


def classify_language(text: str) -> tuple[str, int, int]:
    """Return (language, n_roman_urdu_markers, n_english_markers).

    Rules (applied in order):
      1. no Latin or Arabic-script letters -> 'none' (emoji, numbers, punctuation only)
      2. Arabic-script share of letters >= 0.8 -> 'urdu_script'; 0.2 to 0.8 -> 'mixed'
      3. Latin text: count marker tokens. share = RU / (RU + EN).
         share >= 0.75 -> 'roman_urdu'; share <= 0.25 -> 'english'; otherwise 'mixed';
         no marker tokens at all -> 'undetermined' (names, numbers, single loanwords).
    """
    if not isinstance(text, str):
        return "none", 0, 0
    n_ar = len(ARABIC_RE.findall(text))
    n_lat = len(LATIN_RE.findall(text))
    if n_ar + n_lat == 0:
        return "none", 0, 0
    ar_share = n_ar / (n_ar + n_lat)
    if ar_share >= 0.8:
        return "urdu_script", 0, 0
    tokens = WORD_RE.findall(text.lower())
    ru = sum(t in ROMAN_URDU_MARKERS for t in tokens)
    en = sum(t in ENGLISH_MARKERS for t in tokens)
    if ar_share >= 0.2:
        return "mixed", ru, en
    if ru + en == 0:
        return "undetermined", ru, en
    share = ru / (ru + en)
    if share >= 0.75:
        return "roman_urdu", ru, en
    if share <= 0.25:
        return "english", ru, en
    return "mixed", ru, en


PLACEHOLDER_RE = re.compile(
    r"^(<media omitted>|this message was deleted|you deleted this message|null|"
    r"<this message was edited>|STK-.*\.webp \(file attached\)|https?://\S+)$", re.IGNORECASE)


def main() -> None:
    parse = load_module("stage01", "01_parse.py")
    sent = load_module("stage05", "05_sentiment.py")
    settings = yaml.safe_load(open(J / "config" / "settings.yaml"))

    msgs = pd.read_csv(J / "interim" / "messages.csv")
    msgs["timestamp"] = pd.to_datetime(msgs["timestamp"])
    msgs["msg_id"] = np.arange(len(msgs))
    msgs["phase"] = msgs["phase"].astype("Int64").astype(str)

    # ---- replicate stage 05 input exactly (system rows dropped; join on media_filename)
    df = msgs[msgs["message_type"] != "system"].dropna(subset=["timestamp"]).copy()
    vn = pd.read_csv(J / "processed" / "voice_notes.csv")
    asrun_join = df["media_filename"].astype(str).isin(set(vn["filename"])).sum()

    # ---- link voice notes properly: (prefix, date, WA sequence)
    vn_key = vn["filename"].str.extract(r"^(PTT|AUD)-(\d{8})-WA(\d+)\.opus$", flags=re.I)
    vn = vn.assign(k_pref=vn_key[0].str.upper(), k_date=vn_key[1], k_seq=vn_key[2].astype(float))
    mk = df["message"].astype(str).str.extract(r"^(PTT|AUD)-.*?-WA(\d+)\.opus", flags=re.I)
    df["k_pref"] = mk[0].str.upper()
    df["k_seq"] = mk[1].astype(float)
    df["k_date"] = df["timestamp"].dt.strftime("%Y%m%d")
    lut = vn.drop_duplicates(["k_pref", "k_date", "k_seq"]).set_index(["k_pref", "k_date", "k_seq"])
    def _link(r, shift_days=0):
        if pd.isna(r["k_pref"]):
            return None
        d = (r["timestamp"] + pd.Timedelta(days=shift_days)).strftime("%Y%m%d")
        key = (r["k_pref"], d, r["k_seq"])
        return lut.index.get_loc(key) if key in lut.index else None
    links = []
    for _, r in df.iterrows():
        pos = None
        how = None
        for s in (0, -1, 1):
            pos = _link(r, s)
            if pos is not None:
                how = "same_day" if s == 0 else f"day_shift_{s:+d}"
                break
        links.append((pos, how))
    df["vn_row"] = [p for p, _ in links]
    df["link_rule"] = [h for _, h in links]
    lut_reset = lut.reset_index()
    df["vn_filename"] = df["vn_row"].map(lambda p: lut_reset.at[int(p), "filename"] if p is not None and not pd.isna(p) else None)
    df["transcription_ur"] = df["vn_row"].map(lambda p: lut_reset.at[int(p), "transcription_ur"] if p is not None and not pd.isna(p) else None)
    df["translation_en"] = df["vn_row"].map(lambda p: lut_reset.at[int(p), "translation_en"] if p is not None and not pd.isna(p) else None)

    # ---- cross-check against the engagement agent's shared join (revision/analyses/eng_common.py):
    # its rule is the same-day, PTT-prefix subset of the rule above.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eng_common
    df["vn_filename_eng_common"] = eng_common.reconstruct_vn_filename(df)
    df["linked_by_eng_common"] = df["vn_filename_eng_common"].isin(set(vn["filename"]))
    agree = int((df["linked_by_eng_common"] & (df["vn_filename_eng_common"] == df["vn_filename"])).sum())

    # ---- message type
    case_trigger = re.compile(r"case\s*no|case\s*\.?\s*\d+|🚑", re.IGNORECASE)
    def n_fields(msg: str) -> int:
        n = 0
        for line in str(msg).splitlines():
            m = parse.FIELD_LINE_RE.match(line.lstrip("‎").strip())
            if m and parse._canonical_field(m.group(1)):
                n += 1
        return n
    def mtype(r) -> str:
        if pd.isna(r["sender_code"]):
            # e.g. security-code and encryption notices that 01_parse.py did not recognise
            return "system_unparsed"
        if r["message_type"] == "voice_note":
            return "voice_note"
        if r["message_type"] != "text":
            return "media_placeholder"
        msg = str(r["message"]).strip()
        if PLACEHOLDER_RE.match(msg):
            return "media_placeholder"
        if case_trigger.search(msg) and n_fields(msg) >= 2:
            return "case_report_text"
        return "free_text"
    df["msg_type"] = df.apply(mtype, axis=1)

    # ---- language: text messages classified on their own text; voice notes on the transcript
    lang_src = np.where(df["msg_type"] == "voice_note", df["transcription_ur"], df["message"])
    res = [classify_language(t) for t in lang_src]
    df["language"] = [r[0] for r in res]
    df["ru_markers"] = [r[1] for r in res]
    df["en_markers"] = [r[2] for r in res]
    df.loc[df["msg_type"].isin(["media_placeholder", "system_unparsed"]), "language"] = "none"
    df.loc[(df["msg_type"] == "voice_note") & df["translation_en"].isna(), "language"] = "unlinked_voice_note"

    # ---- text actually analysed
    df["text_asrun"] = df["message"].fillna("").astype(str)
    df["text_linked"] = df["translation_en"].fillna(df["message"]).fillna("").astype(str)

    # ---- lexicon flags (substring rule of stages 05/06), on as-run and linked text
    lex = {
        "urgency": settings["urgency_keywords"],
        "hedging": settings["hedging_keywords"],
        "peer_support": settings["peer_support_keywords"],
        "workaround": settings["workaround_keywords"],
    }
    for comp, kws in settings["toc_component_keywords"].items():
        lex[f"toc_{comp}"] = kws
    for name, kws in lex.items():
        df[f"flag_{name}"] = df["text_linked"].map(lambda t: sent.match_any(t, kws))
        df[f"flag_{name}_asrun"] = df["text_asrun"].map(lambda t: sent.match_any(t, kws))
    df["flag_toc_any"] = df[[c for c in df.columns if c.startswith("flag_toc_") and not c.endswith("_asrun")]].any(axis=1)

    # ---- sentiment (local model, offline)
    from transformers import pipeline
    clf = pipeline("sentiment-analysis", model=sent.MODEL_NAME, tokenizer=sent.MODEL_NAME)
    cache: dict[str, tuple[str, float, float]] = {}
    def score(txt: str):
        if txt in cache:
            return cache[txt]
        if not txt.strip():
            out = ("", np.nan, 0.0)
        else:
            try:
                r = clf(txt[:512])[0]
                out = (r["label"], r["score"], sent.score_to_number(r["label"], r["score"]))
            except Exception:
                out = ("error", np.nan, 0.0)
        cache[txt] = out
        return out
    a = [score(t) for t in df["text_asrun"]]
    l = [score(t) for t in df["text_linked"]]
    df["label_asrun"] = [x[0] for x in a]
    df["score_asrun"] = [x[2] for x in a]
    df["label_linked"] = [x[0] for x in l]
    df["prob_linked"] = [x[1] for x in l]
    df["score_linked"] = [x[2] for x in l]

    # ---- replication check against the archived sentiment_metrics.csv
    arch = pd.read_csv(J / "processed" / "sentiment_metrics.csv")
    arch_daily = arch[arch["kind"] == "rolling_sentiment"].copy()
    arch_daily["timestamp"] = pd.to_datetime(arch_daily["timestamp"])
    arch_daily = arch_daily.set_index("timestamp")["sentiment_daily"]
    rep_daily = df.set_index("timestamp")["score_asrun"].resample("D").mean()
    lnk_daily = df.set_index("timestamp")["score_linked"].resample("D").mean()
    j = pd.concat([arch_daily.rename("archived"), rep_daily.rename("replicated")], axis=1).dropna()
    max_abs = float((j["archived"] - j["replicated"]).abs().max())
    arch_phase = arch[arch["kind"] == "phase_summary"][["phase", "sentiment_mean"]]
    lines = [
        "Replication of archived stage-05 sentiment (data/Jheeng/processed/sentiment_metrics.csv)",
        f"messages scored (non-system): {len(df)}",
        f"voice-note rows whose media_filename matched voice_notes.csv in the archived join: {int(asrun_join)}",
        f"days compared: {len(j)}; max absolute difference in daily mean: {max_abs:.2e}",
        f"correlation archived vs replicated daily mean: {j['archived'].corr(j['replicated']):.6f}",
        "",
        "Phase mean sentiment (archived file / replicated as-run / linked translations):",
    ]
    for p in PHASES:
        av = arch_phase.loc[arch_phase["phase"].astype(str).str.replace('.0', '', regex=False) == p, "sentiment_mean"]
        sub = df[df["phase"] == p]
        lines.append(f"  {p}: {float(av.iloc[0]):.3f} / {sub['score_asrun'].mean():.3f} / {sub['score_linked'].mean():.3f}")
    lines.append(f"  overall: - / {df['score_asrun'].mean():.3f} / {df['score_linked'].mean():.3f}")
    vnm = df["msg_type"] == "voice_note"
    lines += [
        "",
        f"voice notes in chat: {int(vnm.sum())}; linked to a transcript: {int(df.loc[vnm, 'translation_en'].notna().sum())}",
        f"link rule counts: {df.loc[vnm, 'link_rule'].value_counts(dropna=False).to_dict()}",
        f"eng_common.py join (PTT prefix, same day): {int(df.loc[vnm, 'linked_by_eng_common'].sum())} linked; "
        f"identical file to this script's link: {agree}; additional links here: AUD prefix "
        f"{int((vnm & (df['k_pref'] == 'AUD') & df['translation_en'].notna()).sum())}, day-shifted "
        f"{int((vnm & df['link_rule'].astype(str).str.startswith('day_shift')).sum())}",
        f"voice-note placeholder text scored negative (as-run): {(df.loc[vnm, 'label_asrun'] == 'NEGATIVE').mean():.3f}; mean {df.loc[vnm, 'score_asrun'].mean():.3f}",
        f"voice-note translations scored negative (linked): {(df.loc[vnm, 'label_linked'] == 'NEGATIVE').mean():.3f}; mean {df.loc[vnm, 'score_linked'].mean():.3f}",
        f"text (non-placeholder) mean score: {df.loc[df['msg_type'].isin(['free_text','case_report_text']), 'score_asrun'].mean():.3f}",
        f"media placeholder mean score: {df.loc[df['msg_type']=='media_placeholder', 'score_asrun'].mean():.3f}",
    ]
    # rolling 14-day trajectory endpoints, for the narrative
    for nm, ser in (("as-run", rep_daily), ("linked", lnk_daily)):
        r14 = ser.rolling(14, min_periods=1).mean()
        lines.append(f"14-day rolling mean ({nm}): first 60 days mean {r14.iloc[:60].mean():.3f}; "
                     f"last 60 days mean {r14.iloc[-60:].mean():.3f}")
    (OUT / "asrun_replication_check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    keep = ["msg_id", "timestamp", "phase", "sender_code", "message_type", "msg_type", "language",
            "ru_markers", "en_markers", "message", "vn_filename", "link_rule", "transcription_ur",
            "translation_en", "linked_by_eng_common", "label_asrun", "score_asrun", "label_linked", "prob_linked", "score_linked"]
    keep += [c for c in df.columns if c.startswith("flag_")]
    df[keep].to_csv(OUT / "message_level.csv", index=False)

    # ---- composition tables
    comp = (df.groupby(["phase", "msg_type", "language"]).size().rename("n").reset_index())
    comp["pct_of_phase"] = comp["n"] / comp.groupby("phase")["n"].transform("sum") * 100
    tot = df.groupby(["msg_type", "language"]).size().rename("n").reset_index().assign(phase="Overall")
    tot["pct_of_phase"] = tot["n"] / tot["n"].sum() * 100
    pd.concat([comp, tot]).to_csv(OUT / "lang_composition.csv", index=False)

    # language share among analysable content (free text + voice notes), by phase
    flagcols = ["flag_urgency", "flag_hedging", "flag_peer_support", "flag_workaround", "flag_toc_any"]
    analysable = df[df["msg_type"].isin(["free_text", "case_report_text", "voice_note"])]
    rows = []
    for (p, t, lg), sub in pd.concat([analysable, analysable.assign(phase="Overall")]).groupby(["phase", "msg_type", "language"]):
        r = {"phase": p, "msg_type": t, "language": lg, "n": len(sub),
             "mean_score_asrun": sub["score_asrun"].mean(), "mean_score_linked": sub["score_linked"].mean(),
             "pct_negative_linked": (sub["label_linked"] == "NEGATIVE").mean() * 100}
        for c in flagcols:
            r[c.replace("flag_", "pct_")] = sub[c].mean() * 100
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "lang_flags_sentiment.csv", index=False)

    # phase-level means under alternative corpus definitions
    rows = []
    for p, sub in pd.concat([df, df.assign(phase="Overall")]).groupby("phase"):
        txt = sub[sub["msg_type"].isin(["free_text", "case_report_text"])]
        rows.append({
            "phase": p,
            "n_all": len(sub),
            "mean_asrun_all": sub["score_asrun"].mean(),
            "mean_linked_all": sub["score_linked"].mean(),
            "mean_text_only": txt["score_asrun"].mean(),
            "mean_free_text_only": sub.loc[sub["msg_type"] == "free_text", "score_asrun"].mean(),
            "mean_voice_translations": sub.loc[sub["msg_type"] == "voice_note", "score_linked"].mean(),
            "mean_voice_placeholders_asrun": sub.loc[sub["msg_type"] == "voice_note", "score_asrun"].mean(),
            "share_voice_notes_pct": (sub["msg_type"] == "voice_note").mean() * 100,
            "share_placeholders_pct": (sub["msg_type"] == "media_placeholder").mean() * 100,
            "share_case_reports_pct": (sub["msg_type"] == "case_report_text").mean() * 100,
        })
    pd.DataFrame(rows).to_csv(OUT / "lang_sentiment_by_phase.csv", index=False)
    print(f"[val_01] wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
