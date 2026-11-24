"""Step 2 to 4: distribution of response latency, working-hours restriction,
and right-censored time-to-event estimates.

Unit of analysis
  original  : every message paired with the next message from any other sender
              (src/03 rule; a burst of N messages by one sender yields N pairs
              that all share the same reply).
  turn      : consecutive messages by the same sender are collapsed into one
              turn; latency = time from the LAST message of a turn to the first
              message of the next turn (always another sender). Primary unit.
  turn_first: as turn, but timed from the FIRST message of the turn (upper
              bound; sensitivity).

Outputs (data/Jheeng/revision/engagement/):
  02_distribution.csv, 02_median_bootstrap.csv, 02_turns.csv,
  02_workhours.csv, 02_km_threshold.csv, 02_logrank.csv,
  02_by_prompt_sender.csv, fig_latency_ecdf.png, fig_latency_box.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eng_common import (OUT, PHASES, WINDOW_END, build_turns, describe, flag_case_reports,
                        km, km_at, km_median, load_clean, load_raw, logrank,
                        working_minutes)

RNG = np.random.default_rng(20261111)
COLS = {"2024": "#1b6ca8", "2025": "#d1495b", "2026": "#2e8b57"}


def dist_row(x: pd.Series, unit: str, phase: str) -> dict:
    r = {"unit": unit, "phase": phase, **describe(x)}
    x = x.dropna()
    for hrs in (2, 12, 24):
        r[f"share_gt_{hrs}h"] = float((x > hrs * 60).mean()) if len(x) else np.nan
    for m in (15, 60, 240):
        r[f"share_le_{m}min"] = float((x <= m).mean()) if len(x) else np.nan
    return r


def cluster_boot_median(t: pd.DataFrame, col: str, B: int = 2000) -> pd.DataFrame:
    """Percentile CIs for per-phase medians and phase differences, resampling
    calendar days (prompt date) within phase to respect within-day clustering."""
    t = t.dropna(subset=[col]).copy()
    t["day"] = t["end"].dt.normalize()
    groups = {p: [g[col].to_numpy() for _, g in t[t.phase == p].groupby("day")] for p in PHASES}
    est = {p: np.median(np.concatenate(groups[p])) for p in PHASES}
    boots = {p: np.empty(B) for p in PHASES}
    for p in PHASES:
        gs = groups[p]
        k = len(gs)
        for b in range(B):
            idx = RNG.integers(0, k, k)
            boots[p][b] = np.median(np.concatenate([gs[i] for i in idx]))
    rows = []
    for p in PHASES:
        lo, hi = np.percentile(boots[p], [2.5, 97.5])
        rows.append({"contrast": p, "estimate_min": est[p], "ci_lo": lo, "ci_hi": hi,
                     "n_days": len(groups[p])})
    for a, b in (("2025", "2024"), ("2026", "2025"), ("2026", "2024")):
        diff = boots[a] - boots[b]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append({"contrast": f"{a} minus {b}", "estimate_min": est[a] - est[b],
                     "ci_lo": lo, "ci_hi": hi, "n_days": np.nan})
    return pd.DataFrame(rows)


def window_close(ts: pd.Timestamp, b: int) -> pd.Timestamp:
    return ts.normalize() + pd.Timedelta(hours=b)


def main() -> None:
    pd.set_option("display.width", 250)
    df = load_clean()
    df["is_case_report"] = flag_case_reports(df)
    turns = build_turns(df)
    # The final turn has no successor: right-censored at the end of the window.
    turns["event"] = turns["next_start"].notna()
    turns.loc[~turns["event"], "lat_last_min"] = (WINDOW_END - turns["end"]).dt.total_seconds() / 60
    turns.to_csv(OUT / "02_turns.csv", index=False)

    # original message-level pairs (named prompt senders only, as in the published mean)
    orig = pd.read_csv(OUT / "01_latency_pairs_original.csv", parse_dates=["prompt_ts", "reply_ts"])
    orig["phase"] = orig["phase"].astype(str)
    orig = orig[orig["sender_code"].notna()]

    ev = turns[turns["event"]]
    rows = []
    for p in ["Overall"] + PHASES:
        sel = (lambda d: d) if p == "Overall" else (lambda d, p=p: d[d.phase == p])
        rows.append(dist_row(sel(orig)["latency_minutes"], "original_message_pairs", p))
        rows.append(dist_row(sel(ev)["lat_last_min"], "turn_last_to_next", p))
        rows.append(dist_row(sel(ev)["lat_first_min"], "turn_first_to_next", p))
        nb = sel(ev)[~sel(ev)["all_case_report"]]
        rows.append(dist_row(nb["lat_last_min"], "turn_excl_case_report_broadcasts", p))
    dist = pd.DataFrame(rows)
    dist.to_csv(OUT / "02_distribution.csv", index=False)
    print(dist[["unit", "phase", "n", "median", "q1", "q3", "mean", "p90", "share_gt_2h",
                "share_gt_12h", "share_gt_24h", "share_le_15min", "share_le_60min"]].round(2).to_string(index=False))

    boot = cluster_boot_median(ev, "lat_last_min")
    boot.to_csv(OUT / "02_median_bootstrap.csv", index=False)
    print(boot.round(1).to_string(index=False))

    # ---- per prompt sender ----------------------------------------------------
    bys = (ev.groupby(["sender_code", "phase"])["lat_last_min"]
             .agg(n="size", median="median", q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75))
             .reset_index())
    bys.to_csv(OUT / "02_by_prompt_sender.csv", index=False)
    ev2 = ev.assign(grp=np.where(ev.sender_code == "P024", "P024", "other"))
    print(ev2.groupby(["phase", "grp"])["lat_last_min"].agg(["size", "median"]).round(1))

    # ---- working hours --------------------------------------------------------
    wh_rows, lr_rows = [], []
    for a, b in ((6, 22), (8, 20)):
        label = f"{a:02d}:00-{b:02d}:00"
        hrs = turns["end"].dt.hour
        w = turns[(hrs >= a) & (hrs < b)].copy()
        # Rule A, clock-stop: only working-hours minutes elapsed count.
        w["wmin"] = [working_minutes(t0, t1, a, b) if e else np.nan
                     for t0, t1, e in zip(w["end"], w["next_start"], w["event"])]
        # Rule B, same-window censoring: a reply must arrive before the working
        # window closes on the day the prompt was posted; otherwise the prompt is
        # censored at the close of the window.
        close = w["end"].apply(lambda t: window_close(t, b))
        to_close = (close - w["end"]).dt.total_seconds() / 60
        got = w["event"] & (w["next_start"] <= close)
        w["dur_B"] = np.where(got, w["lat_last_min"], to_close)
        w["evt_B"] = got
        for p in ["Overall"] + PHASES:
            s = w if p == "Overall" else w[w.phase == p]
            dA = describe(s["wmin"])
            c = km(s["dur_B"], s["evt_B"])
            wh_rows.append({
                "window": label, "phase": p, "n_prompts": len(s),
                "n_prompts_excluded_outside_window": int(((turns.phase == p) | (p == "Overall")).sum() - len(s)),
                "A_median_working_min": dA.get("median"), "A_q1": dA.get("q1"), "A_q3": dA.get("q3"),
                "A_mean": dA.get("mean"), "A_p90": dA.get("p90"),
                "A_share_le_15": float((s["wmin"] <= 15).mean()),
                "A_share_le_60": float((s["wmin"] <= 60).mean()),
                "A_share_le_240": float((s["wmin"] <= 240).mean()),
                "B_n_replied_same_window": int(s["evt_B"].sum()),
                "B_n_censored_at_close": int((~s["evt_B"]).sum()),
                "B_km_median_min": km_median(c),
                "B_resp_by_15": 1 - km_at(c, 15), "B_resp_by_60": 1 - km_at(c, 60),
                "B_resp_by_240": 1 - km_at(c, 240),
            })
        st, dfree, pv = logrank(w["dur_B"], w["evt_B"], w["phase"])
        lr_rows.append({"analysis": f"working hours {label}, same-window censoring", "groups": "2024/2025/2026",
                        "chi2": st, "df": dfree, "p": pv})
        # Rule A has no censoring; compare distributions with Kruskal-Wallis
        from scipy.stats import kruskal
        kw = kruskal(*[w.loc[w.phase == p, "wmin"].dropna() for p in PHASES])
        lr_rows.append({"analysis": f"working hours {label}, clock-stop (Kruskal-Wallis)", "groups": "2024/2025/2026",
                        "chi2": kw.statistic, "df": 2, "p": kw.pvalue})
    wh = pd.DataFrame(wh_rows)
    wh.to_csv(OUT / "02_workhours.csv", index=False)
    print(wh.round(2).to_string(index=False))

    # ---- threshold censoring (time-to-event) ----------------------------------
    km_rows = []
    curves = {}
    for T in (2, 4, 8):
        Tm = T * 60
        dur = np.minimum(turns["lat_last_min"], Tm)
        evt = turns["event"] & (turns["lat_last_min"] <= Tm)
        tt = turns.assign(dur=dur, evt=evt)
        for p in ["Overall"] + PHASES:
            s = tt if p == "Overall" else tt[tt.phase == p]
            c = km(s["dur"], s["evt"])
            curves[(T, p)] = c
            km_rows.append({"censor_at_h": T, "phase": p, "n": len(s), "n_events": int(s["evt"].sum()),
                            "n_censored": int((~s["evt"]).sum()),
                            "km_median_min": km_median(c),
                            "resp_by_15min": 1 - km_at(c, 15), "resp_by_1h": 1 - km_at(c, 60),
                            "resp_by_4h": (1 - km_at(c, 240)) if Tm >= 240 else np.nan})
        st, dfree, pv = logrank(tt["dur"], tt["evt"], tt["phase"])
        lr_rows.append({"analysis": f"turn latency censored at {T} h", "groups": "2024/2025/2026",
                        "chi2": st, "df": dfree, "p": pv})
        for a_, b_ in (("2024", "2025"), ("2025", "2026"), ("2024", "2026")):
            s2 = tt[tt.phase.isin([a_, b_])]
            st, dfree, pv = logrank(s2["dur"], s2["evt"], s2["phase"])
            lr_rows.append({"analysis": f"turn latency censored at {T} h", "groups": f"{a_} vs {b_}",
                            "chi2": st, "df": dfree, "p": pv})
    kmt = pd.DataFrame(km_rows)
    kmt.to_csv(OUT / "02_km_threshold.csv", index=False)
    lr = pd.DataFrame(lr_rows)
    lr.to_csv(OUT / "02_logrank.csv", index=False)
    print(kmt.round(3).to_string(index=False))
    print(lr.round(4).to_string(index=False))

    # ---- figures --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (data, col, title) in zip(axes, [
            (orig, "latency_minutes", "A. Original rule: every message to next other sender"),
            (ev, "lat_last_min", "B. Turn transitions (primary)")]):
        for p in PHASES:
            x = np.sort(data.loc[data.phase == p, col].clip(lower=0.5).to_numpy())
            y = np.arange(1, len(x) + 1) / len(x)
            ax.step(x, y, where="post", color=COLS[p], label=f"{p} (n={len(x)})")
        for v, lab in ((15, "15 min"), (60, "1 h"), (240, "4 h"), (1440, "24 h")):
            ax.axvline(v, color="0.8", lw=0.8, zorder=0)
            ax.text(v, 0.02, lab, rotation=90, fontsize=7, color="0.4", ha="right")
        ax.set_xscale("log")
        ax.set_xlabel("Minutes to next message from another sender (log; 0 plotted at 0.5)", fontsize=8)
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(fontsize=8, loc="lower right", frameon=False)
        ax.grid(False)
    axes[0].set_ylabel("Cumulative proportion")
    fig.tight_layout()
    fig.savefig(OUT / "fig_latency_ecdf.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [ev.loc[ev.phase == p, "lat_last_min"].clip(lower=0.5) for p in PHASES]
    bp = ax.boxplot(data, whis=(10, 90), showfliers=False, patch_artist=True)
    for patch, p in zip(bp["boxes"], PHASES):
        patch.set_facecolor(COLS[p]); patch.set_alpha(0.35)
    for i, (p, x) in enumerate(zip(PHASES, data), start=1):
        ax.scatter(np.full(len(x), i) + RNG.uniform(-0.18, 0.18, len(x)), x, s=4, color=COLS[p], alpha=0.35)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3], [f"{p}\n(n={len(x)})" for p, x in zip(PHASES, data)])
    ax.set_ylabel("Turn-transition latency, minutes (log)")
    ax.set_title("Box: median and IQR; whiskers: 10th to 90th percentile", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_latency_box.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for p in PHASES:
        c = curves[(8, p)]
        ax.step(c["time"] / 60, 1 - c["surv"], where="post", color=COLS[p], label=p)
    ax.set_xlabel("Hours since end of prompting turn (censored at 8 h)")
    ax.set_ylabel("Cumulative proportion answered (1 minus KM)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_latency_km.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
