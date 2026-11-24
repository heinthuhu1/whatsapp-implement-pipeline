"""Revision R1 #13: revised clinical profile of the extracted case reports (Jheeng).

Reads ONLY archived Jheeng data. Writes ONLY to data/Jheeng/revision/clinical_repro/.
Nothing in src/ is modified; stage-01 functions are imported read-only.

Outputs (all local, gitignored)
  clin_original_reproduction.txt   reproduces the manuscript numbers (639; 51/30/19; 101/45/15/15)
  clin_sex.csv, clin_age.csv       sex and age-group distributions, overall and by phase
  clin_gender_field_mapping.csv    every distinct gender-field value -> sex, age group, rule
  complaint_mapping.csv            every distinct complaint string -> spelling fix -> category
  clin_complaints.csv              primary-category distribution, overall and by phase
  clin_complaints_anymention.csv   multi-label (any mention) counts per category
  clin_completeness.csv            per-field completeness: as extracted, and label present in source
  clin_duplicates.csv              duplicate case-number groups and how each was classified
  clin_summary.txt                 headline numbers used in FINDINGS.md
  fig5_case_profile_revised.png    revised Figure 5
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
J = ROOT / "data" / "Jheeng"
OUT = J / "revision" / "clinical_repro"
OUT.mkdir(parents=True, exist_ok=True)
PHASES = ["2024", "2025", "2026"]
WIN_START, WIN_END = pd.Timestamp("2024-09-22"), pd.Timestamp("2026-04-11")

spec = importlib.util.spec_from_file_location("stage01", ROOT / "src" / "01_parse.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def norm(s) -> str | None:
    if pd.isna(s):
        return None
    s = re.sub(r"\s+", " ", str(s)).strip().lower()
    return s or None


# =============================================================== sex and age
def sex_age(gender: str | None, raw: str) -> tuple[str, str, str]:
    """Return (sex, age_group, rule). The template has no age field; age group is
    inferred only from explicit cues in the Gender field or an explicit 'Age:' line."""
    age_m = re.search(r"(?im)^\W*age\s*[:.]\s*(\d{1,3})", raw or "")
    g = norm(gender)
    if g is None:
        sex, age, rule = "missing", "not recorded", "gender field blank"
    else:
        is_f = bool(re.search(r"\bfemale\b|\bgirl\b|\bwoman\b", g))
        is_m = (not is_f) and bool(re.search(r"\bmale\b|\bmal\b|\bman\b|\bboy\b", g))
        child = bool(re.search(r"child|children|newborn|baby|infant|\bboy\b|\bgirl\b", g))
        old = bool(re.search(r"\bold\b", g))
        sex = "female" if is_f else "male" if is_m else "not stated"
        if child:
            age, rule = "child (paediatric)", "child term in Gender field"
        elif old:
            age, rule = "adult", "'old' in Gender field"
        else:
            age, rule = "not recorded", "sex only; template has no age field"
        if sex == "not stated":
            rule += "; sex not stated"
    if age_m:
        a = int(age_m.group(1))
        age = "child (paediatric)" if a < 18 else "adult"
        rule = f"explicit Age line ({'<18' if a < 18 else '>=18'})"
    return sex, age, rule


# =============================================================== complaints
SPELLING = [  # (pattern, replacement) applied to the normalised complaint string, in order
    (r"\bb\s*\.?\s*p\b\.?", "bp"), (r"\bt\.b\b", "tb"),
    (r"\bapendix\b", "appendix"), (r"\bstomac\b|\bstmoac\b|\bctomac\b", "stomach"),
    (r"\bmoshan\b|\bmosion\b|\bmostion\b|\bmossion\b", "motion"),
    (r"\bcanser\b", "cancer"), (r"\bkidny\b|\bkidni\b", "kidney"),
    (r"\bfracher\b|\bfachar\b|\bfacture\b|\bfracpure\b|\bfixture\b|\bfarghure\b", "fracture"),
    (r"\basthama\b|\basdama\b", "asthma"), (r"\bnamoni(ya)?\b", "pneumonia"),
    (r"\btyphaide\b", "typhoid"), (r"\bdelevry\b|\b0peration\b", lambda m: "delivery" if m.group(0) == "delevry" else "operation"),
    (r"\bbleding\b|\bblooding\b", "bleeding"), (r"\banjury\b|\binjured\b", "injury"),
    (r"\bhearth\b", "heart"), (r"\bheat attack\b", "heart attack"), (r"\belergy\b", "allergy"),
    (r"\bsuger\b", "sugar"), (r"\bopration\b|\boperaction\b|\boprita\b", "operation"),
    (r"\barina\b|\bharnia\b", "hernia"), (r"\baram\b", "arm"), (r"\bfingre\b", "finger"),
    (r"\bieg\b", "leg"), (r"\balbo\b", "elbow"), (r"\bbrian\b", "brain"),
    (r"\bfalaga\b|\bfallag\b|\bparalysies\b|\bparalize\b|\bparalytic\b", "paralysis"),
    (r"\bwarmting\b", "vomiting"), (r"\bv omitting\b", "vomiting"), (r"\bconstation\b", "constipation"),
    (r"\btarom\b|\btromo\b", "trauma"), (r"\bsayko\b", "psychiatric"), (r"\bkhasra\b", "measles"),
    (r"\bgynee\b", "gynae"), (r"\bmantal stat\w*\b", "mental status"), (r"\bnosi\b|\bnoses\b", "nose"),
    (r"\bkidney p\b", "kidney problem"), (r"\bbandge\b", "bandage"), (r"\bmadical\b", "medical"),
    (r"\bad\b", "and"), (r"\bd delivery\b", "delivery"),
]

# Ordered category rules: the FIRST matching rule gives the primary category.
CATEGORY_RULES = [
    ("Death / body transport", "", r"dead body|death body|died body"),
    ("Obstetric and gynaecological", "", r"deliver|pregnan|gyna|labou?r"),
    ("Injury", "Fall", r"\bfell\b|fallen|\bfall\b"),
    ("Injury", "Bite or sting", r"bite|black bee|sting"),
    ("Injury", "Burn or electrical", r"burn|electric|electroc"),
    ("Injury", "Fracture, mechanism not recorded", r"fracture"),
    ("Injury", "Other injury, mechanism not recorded", r"injur|trauma|internal bleeding|bandage|^and injury$"),
    ("Stroke and paralysis", "", r"stroke|paralysis"),
    ("Cardiovascular and blood pressure", "", r"heart|chest pain|\bbp\b|blood pressure"),
    ("Respiratory", "", r"asthma|lung|pneumonia|oxygen|\btb\b"),
    ("Fever and infection", "", r"fever|typhoid|measles|chickenpox|infection"),
    ("Gastrointestinal and abdominal", "", r"motion|vomit|stomach|abdom|appendix|food poison|dysentery|gallbladder|constipation"),
    ("Renal and urological", "", r"kidney|dialysis|bladder|urine"),
    ("Neurological and mental health", "", r"brain|unconscious|syncope|faint|mental status|psychiatric"),
    ("Metabolic (diabetes / glucose)", "", r"sugar|diabet"),
    ("Oncology", "", r"cancer|tumou?r"),
    ("Musculoskeletal pain (non-traumatic)", "", r"back|sciatica|muscular|orthop|leg pain|leg problem"),
    ("Allergy, skin and drug reaction", "", r"allerg|skin|acne|pimple|reaction|re action"),
    ("Eye, ear, nose and throat", "", r"\beye|\bnose\b|foreign object"),
    ("Surgical, unspecified", "", r"hernia|operation"),
    ("Non-emergency transport", "", r"appointment"),
]
UNCLASSIFIABLE = "Unclassifiable"
MISSING = "Missing / blank"


def fix_spelling(s: str) -> str:
    out = s.replace("&", " and ").replace(",", " , ")
    for pat, rep in SPELLING:
        out = re.sub(pat, rep, out)
    return re.sub(r"\s+", " ", out).strip(" .,\"'")


def categorise(s: str | None) -> tuple[str, str, str, list[str]]:
    if s is None or not isinstance(s, str):
        return MISSING, "", "complaint field blank or absent", []
    fixed = fix_spelling(s)
    hits = [(c, sc, pat) for c, sc, pat in CATEGORY_RULES if re.search(pat, fixed)]
    if not hits:
        return UNCLASSIFIABLE, "", "no rule matched (non-specific or unrecognised term)", []
    cat, sub, pat = hits[0]
    allcats = sorted({c for c, _, _ in hits})
    return cat, sub, f"/{pat}/", allcats


# =============================================================== extended label rules (completeness audit)
EXT_LABELS = {
    "date": r"^\W*date\b",
    "gender": r"^\W*gender\b",
    "initial_complaint": r"^\W*(initial\s*complaint|complaint)\b",
    "location": r"^\W*(pickup\s*)?location\b(?!\s*was)(?!\s*transported)",
    "time_patient_reached": r"^\W*(time\s*patien\w*|pickup\s*time)",
    "hospital": r"^\W*(name\s*(of\s*)?hospital|hospital\s*transported|transported\s*to|destination|name\s*of\s*location\s*transported)",
    "time_hospital_reached": r"^\W*(time\s*(hospital|home|location)|hospital\s*time|arrival\s*time)",
    "first_aid": r"^\W*first\s*aid\b",
    "emp_personnel": r"^\W*emp\s*person",
}
LABEL_ONLY = {  # what remains after the label when no value was typed
    "time_patient_reached": r"^\W*(time\s*patien\w*(\s*reached)?|pickup\s*time)\W*$",
    "time_hospital_reached": r"^\W*(time\s*(hospital|home|location)(\s*was)?(\s*reached)?|hospital\s*time\s*reached|arrival\s*time)\W*$",
    "hospital": r"^\W*(name\s*(of\s*)?hospital(\s*transported\s*to)?|transported\s*to|destination)\W*$",
}


def label_status(raw: str, field: str) -> str:
    """'value' if a label line with some value is present, 'label_only' if the label is
    present but empty, 'absent' if no label line is present."""
    status = "absent"
    for line in str(raw).splitlines():
        l = line.lstrip("‎").strip()
        if re.match(EXT_LABELS[field], l, re.I):
            empty = field in LABEL_ONLY and re.match(LABEL_ONLY[field], l, re.I)
            if not empty and field not in LABEL_ONLY:
                empty = re.match(EXT_LABELS[field] + r"[^:;.]*[:;.]?\s*$", l, re.I)
            if not empty:
                return "value"
            status = "label_only"
    return status


def parse_incident_date(s) -> pd.Timestamp:
    if pd.isna(s):
        return pd.NaT
    t = re.sub(r"[^0-9]+", "/", str(s)).strip("/")
    parts = t.split("/")
    if len(parts) != 3:
        return pd.NaT
    d, m, y = parts
    if len(y) == 2:
        y = "20" + y
    try:
        return pd.Timestamp(int(y), int(m), int(d))
    except Exception:
        return pd.NaT


def pct(n, d):
    return round(100 * n / d, 1) if d else np.nan


def dist_table(df: pd.DataFrame, col: str, order: list[str]) -> pd.DataFrame:
    rows = []
    for ph, sub in [("Overall", df)] + [(p, df[df["phase"] == p]) for p in PHASES]:
        n = len(sub)
        recorded = sub[~sub[col].isin(["missing", "not stated", "not recorded"])]
        for cat in order:
            k = int((sub[col] == cat).sum())
            rows.append({"phase": ph, "category": cat, "n": k, "pct_of_all_records": pct(k, n),
                         "pct_of_recorded": pct(k, len(recorded)) if cat in set(recorded[col]) else np.nan,
                         "denominator_all": n, "denominator_recorded": len(recorded)})
    return pd.DataFrame(rows)


def main() -> None:
    cr = pd.read_csv(J / "interim" / "case_reports.csv", dtype={"case_no": str})
    cr["phase"] = cr["phase"].astype(str)
    cr["timestamp"] = pd.to_datetime(cr["timestamp"])
    N = len(cr)
    summ: list[str] = []

    # ---------------- reproduce the manuscript numbers
    raw_top = cr["initial_complaint"].value_counts().head(10)
    g = cr["gender"].str.lower().str.strip().replace(
        {"male child": "child", "children": "child", "mal": "male", "old female": "female"})
    top4 = g.value_counts().head(4)
    rep = [f"records: {N}",
           "original fig5 top-10 complaints (raw strings, committed code): " + "; ".join(f"{k}={v}" for k, v in raw_top.items()),
           "original fig5 gender pie (top 4 after mapping; NaN and 5 minor variants dropped): "
           + "; ".join(f"{k}={v} ({100*v/top4.sum():.1f}%)" for k, v in top4.items())
           + f"; pie denominator={top4.sum()} of {N}"]
    norm_top = cr["initial_complaint"].map(norm).value_counts().head(10)
    rep.append("current src/08 (case/whitespace normalised) top-10: " + "; ".join(f"{k}={v}" for k, v in norm_top.items()))
    (OUT / "clin_original_reproduction.txt").write_text("\n".join(rep) + "\n")
    summ += rep

    # ---------------- sex and age
    sa = [sex_age(gv, rm) for gv, rm in zip(cr["gender"], cr["raw_message"])]
    cr["sex"] = [x[0] for x in sa]
    cr["age_group"] = [x[1] for x in sa]
    cr["sex_age_rule"] = [x[2] for x in sa]
    gm = (cr.assign(gender_normalised=cr["gender"].map(norm).fillna("<blank>"))
            .groupby(["gender_normalised", "sex", "age_group", "sex_age_rule"]).size()
            .rename("n").reset_index().sort_values("n", ascending=False))
    gm.to_csv(OUT / "clin_gender_field_mapping.csv", index=False)
    sex_t = dist_table(cr, "sex", ["female", "male", "not stated", "missing"])
    age_t = dist_table(cr, "age_group", ["child (paediatric)", "adult", "not recorded"])
    sex_t.to_csv(OUT / "clin_sex.csv", index=False)
    age_t.to_csv(OUT / "clin_age.csv", index=False)

    # ---------------- complaints
    cr["complaint_norm"] = cr["initial_complaint"].map(norm)
    cats = [categorise(s) for s in cr["complaint_norm"]]
    cr["complaint_category"] = [c[0] for c in cats]
    cr["complaint_subcategory"] = [c[1] for c in cats]
    cr["complaint_all_categories"] = ["|".join(c[3]) for c in cats]
    mp = (cr.groupby("complaint_norm", dropna=False).size().rename("n").reset_index())
    mp["spelling_normalised"] = mp["complaint_norm"].map(lambda s: fix_spelling(s) if isinstance(s, str) else None)
    mp["spelling_changed"] = mp["spelling_normalised"] != mp["complaint_norm"]
    mc = mp["complaint_norm"].map(lambda s: categorise(s if isinstance(s, str) else None))
    mp["primary_category"] = [c[0] for c in mc]
    mp["subcategory"] = [c[1] for c in mc]
    mp["rule_matched"] = [c[2] for c in mc]
    mp["all_categories_mentioned"] = ["|".join(c[3]) for c in mc]
    mp = mp.sort_values(["primary_category", "n"], ascending=[True, False])
    mp.to_csv(OUT / "complaint_mapping.csv", index=False)

    cat_order = list(dict.fromkeys([c for c, _, _ in CATEGORY_RULES])) + [UNCLASSIFIABLE, MISSING]
    rows = []
    for ph, sub in [("Overall", cr)] + [(p, cr[cr["phase"] == p]) for p in PHASES]:
        for c in cat_order:
            k = int((sub["complaint_category"] == c).sum())
            rows.append({"phase": ph, "category": c, "subcategory": "", "n": k, "pct": pct(k, len(sub)), "denominator": len(sub)})
            if c == "Injury":
                for s in [sc for cc, sc, _ in CATEGORY_RULES if cc == "Injury"]:
                    k2 = int(((sub["complaint_category"] == c) & (sub["complaint_subcategory"] == s)).sum())
                    rows.append({"phase": ph, "category": c, "subcategory": s, "n": k2, "pct": pct(k2, len(sub)), "denominator": len(sub)})
    comp_t = pd.DataFrame(rows)
    comp_t.to_csv(OUT / "clin_complaints.csv", index=False)
    am = []
    for c in cat_order[:-2]:
        k = int(cr["complaint_all_categories"].str.split("|").map(lambda L: c in L).sum())
        am.append({"category": c, "n_any_mention": k, "pct": pct(k, N)})
    pd.DataFrame(am).to_csv(OUT / "clin_complaints_anymention.csv", index=False)

    # ---------------- completeness
    fields = ["case_no", "date", "gender", "initial_complaint", "location", "time_patient_reached",
              "hospital", "time_hospital_reached", "first_aid", "emp_personnel"]
    rows = []
    for ph, sub in [("Overall", cr)] + [(p, cr[cr["phase"] == p]) for p in PHASES]:
        for f in fields:
            ext = int(sub[f].notna().sum())
            r = {"phase": ph, "field": f, "n_records": len(sub), "extracted_n": ext, "extracted_pct": pct(ext, len(sub))}
            if f != "case_no":
                st = sub["raw_message"].map(lambda m: label_status(m, f))
                r["source_value_present_n"] = int((st == "value").sum())
                r["source_value_present_pct"] = pct(r["source_value_present_n"], len(sub))
                r["source_label_empty_n"] = int((st == "label_only").sum())
                r["source_label_absent_n"] = int((st == "absent").sum())
            rows.append(r)
    compl = pd.DataFrame(rows)
    compl.to_csv(OUT / "clin_completeness.csv", index=False)

    # ---------------- duplicates (re-run extraction message by message, which bypasses de-duplication)
    msgs = pd.read_csv(J / "interim" / "messages.csv")
    msgs["timestamp"] = pd.to_datetime(msgs["timestamp"])
    recs = []
    for i in range(len(msgs)):
        r = P.parse_case_reports(msgs.iloc[[i]])
        if len(r):
            recs.append(r)
    pre = pd.concat(recs, ignore_index=True)
    pre["case_no"] = pre["case_no"].astype(str)
    pre["inc_date"] = pre["date"].map(parse_incident_date)
    groups = []
    for cn, sub in pre.groupby("case_no"):
        if len(sub) < 2:
            continue
        key = sub[["date", "gender", "initial_complaint"]].map(norm).astype(str)
        span_days = (sub["timestamp"].max() - sub["timestamp"].min()).total_seconds() / 86400
        d = sub["inc_date"].dropna()
        inc_span = (d.max() - d.min()).days if len(d) > 1 else 0
        sexes = {sex_age(x, "")[0] for x in sub["gender"]} - {"not stated", "missing"}
        ccats = {categorise(norm(x))[0] for x in sub["initial_complaint"]} - {MISSING}
        if key.nunique().max() == 1:
            cls = "exact repost (date, gender, complaint identical)"
        elif (inc_span > 1 or span_days > 2) and (len(sexes) > 1 or len(ccats) > 1):
            cls = "probable distinct cases sharing a number"
        else:
            cls = "correction or partial repost of the same case"
        groups.append({"case_no": cn, "n_records": len(sub), "n_collapsed": len(sub) - 1,
                       "posting_span_days": round(span_days, 1), "incident_date_span_days": inc_span,
                       "classification": cls})
    dup = pd.DataFrame(groups)
    dup.to_csv(OUT / "clin_duplicates.csv", index=False)
    star = cr["case_no"][~cr["case_no"].str.fullmatch(r"\d+")]
    star_clash = [s for s in star if re.sub(r"\D", "", s) in set(cr["case_no"])]

    # ---------------- window checks
    cr["inc_date"] = cr["date"].map(parse_incident_date)
    out_ts = int(((cr["timestamp"] < WIN_START) | (cr["timestamp"] > WIN_END + pd.Timedelta(days=1))).sum())
    inc_before = int((cr["inc_date"] < WIN_START).sum())
    inc_after = int((cr["inc_date"] > WIN_END).sum())
    inc_unparsed = int(cr["date"].notna().sum() - cr["inc_date"].notna().sum())
    lag = (cr["timestamp"].dt.normalize() - cr["inc_date"]).dt.days
    implaus = int(((lag < -1) | (lag > 60)).sum())

    nums = pd.to_numeric(cr["case_no"].str.replace(r"\D", "", regex=True), errors="coerce")
    missing_numbers = sorted(set(range(1, int(nums.max()) + 1)) - set(nums.dropna().astype(int)))

    summ += [
        "",
        f"pre-dedup records: {len(pre)}; after dedup: {N}; collapsed: {len(pre) - N} across {len(dup)} case numbers",
        "dedup classes (groups / collapsed records): " + "; ".join(
            f"{k}: {len(v)} / {int(v['n_collapsed'].sum())}" for k, v in dup.groupby("classification")),
        f"non-numeric case numbers: {list(star)}; of which also present as plain number: {star_clash}",
        f"case-number range 1..{int(nums.max())}; numbers absent from the extracted set: {len(missing_numbers)}",
        f"records with message timestamp outside window: {out_ts}",
        f"incident date (Date field): parsed {int(cr['inc_date'].notna().sum())}, blank {int(cr['date'].isna().sum())}, unparseable {inc_unparsed}; "
        f"before window start {inc_before}; after window end {inc_after}; posting lag <-1 or >60 days: {implaus}",
        "",
        "SEX (overall): " + "; ".join(f"{r.category}={r.n} ({r.pct_of_all_records}%)" for r in sex_t[sex_t.phase == 'Overall'].itertuples()),
        "AGE (overall): " + "; ".join(f"{r.category}={r.n} ({r.pct_of_all_records}%)" for r in age_t[age_t.phase == 'Overall'].itertuples()),
        "COMPLAINT categories (overall): " + "; ".join(f"{r.category}{' / ' + r.subcategory if r.subcategory else ''}={r.n} ({r.pct}%)" for r in comp_t[comp_t.phase == 'Overall'].itertuples()),
        f"distinct complaint strings (case/whitespace normalised): {cr['complaint_norm'].nunique()}; changed by spelling rules: {int(mp['spelling_changed'].sum())}",
        f"emp_personnel field non-empty in {int(cr['emp_personnel'].notna().sum())} records (contains staff first names not caught by redaction; see caveats)",
    ]
    (OUT / "clin_summary.txt").write_text("\n".join(summ) + "\n")
    print("\n".join(summ))

    # ---------------- revised Figure 5
    BLUE, GREY = "#2a6f97", "#b0b7bf"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1], wspace=1.05, hspace=0.45)

    def hbar(ax, labels, values, colors, title, xlabel, denom):
        y = np.arange(len(labels))[::-1]
        ax.barh(y, values, color=colors, height=0.68)
        for yi, v in zip(y, values):
            ax.text(v + max(values) * 0.015, yi, f"{v} ({100 * v / denom:.1f}%)", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, max(values) * 1.32)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)

    o = sex_t[sex_t.phase == "Overall"]
    ax = fig.add_subplot(gs[0, 0])
    hbar(ax, ["Female", "Male", "Not stated (child only)", "Missing"], o["n"].tolist(),
         [BLUE, BLUE, GREY, GREY], f"A  Sex (n={N})", "Records", N)
    o = age_t[age_t.phase == "Overall"]
    ax = fig.add_subplot(gs[1, 0])
    hbar(ax, ["Child (explicit)", "Adult (explicit)", "Not recorded"], o["n"].tolist(),
         [BLUE, BLUE, GREY], f"B  Age group (n={N})", "Records", N)

    ax = fig.add_subplot(gs[:, 1])
    o = comp_t[(comp_t.phase == "Overall")].copy()
    o["label"] = np.where(o["subcategory"] != "", "› " + o["subcategory"], o["category"])
    main = o[(o["subcategory"] == "") & ~o["category"].isin([UNCLASSIFIABLE, MISSING])].sort_values("n", ascending=False)
    seq = []
    for _, r in main.iterrows():
        seq.append(r)
        if r["category"] == "Injury":
            subs = o[(o["category"] == "Injury") & (o["subcategory"] != "")].sort_values("n", ascending=False)
            seq += [s for _, s in subs.iterrows() if s["n"] > 0]
    seq += [o[(o["category"] == c) & (o["subcategory"] == "")].iloc[0] for c in (UNCLASSIFIABLE, MISSING)]
    labels = [s["label"] for s in seq]
    vals = [int(s["n"]) for s in seq]
    cols = [GREY if s["category"] in (UNCLASSIFIABLE, MISSING) else ("#7fb3d5" if s["subcategory"] else BLUE) for s in seq]
    hbar(ax, labels, vals, cols, f"C  Presenting complaint, primary category (n={N})", "Records", N)
    fig.suptitle("Clinical profile of extracted case reports, Jheeng, Sep 2024 to Apr 2026",
                 fontweight="bold", fontsize=11, x=0.02, ha="left")
    fig.text(0.02, -0.04, "Percentages use all 639 de-duplicated records as denominator. Grey bars: missing, not stated, or unclassifiable.\n"
             "Light bars marked › are injury sub-groups (mechanism where recorded). The case-report template has no age field: age group is\n"
             "inferred only from explicit child or age terms; male and female entries without such terms are counted as age not recorded.", fontsize=7.5)
    fig.savefig(OUT / "fig5_case_profile_revised.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[clin_01] wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
