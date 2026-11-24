"""Step 5: conversational sessions.

A session is a maximal run of messages in which every consecutive gap is
strictly shorter than a threshold T (T = 30, 60, 120 min; a gap >= T opens a
new session). Sessions are assigned to the phase of their first message.

Threshold justification: the distribution of log10 inter-message gaps (gaps
> 0 min; timestamps have one-minute resolution) is summarised with a kernel
density estimate and a Gaussian mixture fitted by EM (2 and 3 components, BIC; 4 components degenerated).
For each candidate T we report the share of the 'conversational' component
that T retains inside sessions and the share of the 'idle' component it
wrongly retains.

Outputs (data/Jheeng/revision/engagement/):
  03_sessions_summary.csv, 03_sessions.csv (one row per session, T = 60),
  03_gap_mixture.csv, 03_threshold_curve.csv, fig_gap_distribution.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm

from eng_common import OUT, PHASES, build_turns, describe, load_clean

THRESHOLDS = (30, 60, 120)
DEFAULT = 60


def em_gmm(x: np.ndarray, k: int, iters: int = 1000):
    mu = np.quantile(x, np.linspace(0.15, 0.85, k))
    sd = np.full(k, x.std() / k)
    w = np.full(k, 1 / k)
    for _ in range(iters):
        p = np.array([w[j] * norm.pdf(x, mu[j], sd[j]) for j in range(k)])
        r = p / p.sum(0)
        w = r.mean(1)
        mu = (r * x).sum(1) / r.sum(1)
        sd = np.sqrt((r * (x - mu[:, None]) ** 2).sum(1) / r.sum(1))
    ll = np.log(np.array([w[j] * norm.pdf(x, mu[j], sd[j]) for j in range(k)]).sum(0)).sum()
    bic = -2 * ll + (3 * k - 1) * np.log(len(x))
    order = np.argsort(mu)
    return w[order], mu[order], sd[order], bic


def sessionise(df: pd.DataFrame, T: int) -> pd.DataFrame:
    d = df.copy()
    gap = d["timestamp"].diff().dt.total_seconds() / 60
    d["session_id"] = (gap.isna() | (gap >= T)).cumsum() - 1
    return d


def session_table(d: pd.DataFrame) -> pd.DataFrame:
    s = d.groupby("session_id").agg(
        phase=("phase", "first"), start=("timestamp", "first"), end=("timestamp", "last"),
        n_msgs=("timestamp", "size"), n_senders=("sender_code", "nunique"),
        initiator=("sender_code", "first"))
    s["duration_min"] = (s["end"] - s["start"]).dt.total_seconds() / 60
    return s.reset_index()


def within_session_latency(d: pd.DataFrame) -> pd.DataFrame:
    """Turn transitions whose two turns lie in the same session."""
    out = []
    for sid, g in d.groupby("session_id"):
        if g["sender_code"].nunique() < 2:
            continue
        t = build_turns(g)
        t = t[t["next_start"].notna()]
        out.append(t.assign(session_id=sid))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> None:
    pd.set_option("display.width", 250)
    df = load_clean()
    gaps = (df["timestamp"].diff().dt.total_seconds() / 60).dropna()
    pos = gaps[gaps > 0]
    x = np.log10(pos.to_numpy())

    # ---- mixture model of log gaps -----------------------------------------
    fits = {k: em_gmm(x, k) for k in (2, 3)}
    mix_rows = []
    for k, (w, mu, sd, bic) in fits.items():
        for j in range(k):
            mix_rows.append({"k": k, "component": j + 1, "weight": w[j],
                             "median_gap_min": 10 ** mu[j], "sd_log10": sd[j], "BIC": bic})
    mix = pd.DataFrame(mix_rows)
    best_k = int(mix.loc[mix.BIC.idxmin(), "k"])
    w, mu, sd, _ = fits[3]  # interpretable: burst / conversational / idle
    conv, idle = 1, 2
    curve_rows = []
    for T in (5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 1440):
        n_sessions = int((gaps >= T).sum()) + 1
        lt = np.log10(T)
        curve_rows.append({
            "threshold_min": T, "n_sessions": n_sessions,
            "share_all_gaps_below": float((gaps < T).mean()),
            "conversational_component_retained": norm.cdf(lt, mu[conv], sd[conv]),
            "idle_component_wrongly_retained": norm.cdf(lt, mu[idle], sd[idle]),
        })
    curve = pd.DataFrame(curve_rows)
    # posterior crossing between conversational and idle components
    grid = np.linspace(mu[conv], mu[idle], 5000)
    diff = w[conv] * norm.pdf(grid, mu[conv], sd[conv]) - w[idle] * norm.pdf(grid, mu[idle], sd[idle])
    cross = 10 ** grid[np.where(np.diff(np.sign(diff)))[0][0]]
    kde = gaussian_kde(x, bw_method=0.2)
    xs = np.linspace(0, 4.2, 600)
    ys = kde(xs)
    troughs = [10 ** xs[i] for i in range(1, len(xs) - 1) if ys[i] < ys[i - 1] and ys[i] < ys[i + 1]]
    peaks = [10 ** xs[i] for i in range(1, len(xs) - 1) if ys[i] > ys[i - 1] and ys[i] > ys[i + 1]]
    mix["best_k_by_BIC"] = best_k
    mix["kde_peaks_min"] = ";".join(f"{p:.0f}" for p in peaks)
    mix["kde_troughs_min"] = ";".join(f"{t:.0f}" for t in troughs)
    mix["conv_idle_posterior_crossing_min"] = cross
    mix["n_zero_gaps"] = int((gaps == 0).sum())
    mix["n_gaps"] = int(len(gaps))
    mix.to_csv(OUT / "03_gap_mixture.csv", index=False)
    curve.to_csv(OUT / "03_threshold_curve.csv", index=False)
    print(mix.round(3).to_string(index=False))
    print(curve.round(3).to_string(index=False))
    print("KDE peaks", peaks, "troughs", troughs, "crossing", cross)

    # ---- sessions -----------------------------------------------------------
    rows = []
    for T in THRESHOLDS:
        d = sessionise(df, T)
        s = session_table(d)
        wl = within_session_latency(d)
        if T == DEFAULT:
            s.drop(columns=["initiator"]).to_csv(OUT / "03_sessions.csv", index=False)
        for p in ["Overall"] + PHASES:
            ss = s if p == "Overall" else s[s.phase == p]
            ww = wl if p == "Overall" else wl[wl.phase == p]
            dl = describe(ww["lat_last_min"]) if len(ww) else {"n": 0}
            active_days = (d if p == "Overall" else d[d.phase == p])["timestamp"].dt.normalize().nunique()
            rows.append({
                "threshold_min": T, "phase": p, "n_sessions": len(ss),
                "sessions_per_active_day": len(ss) / active_days,
                "median_msgs": ss["n_msgs"].median(),
                "iqr_msgs": f"{ss['n_msgs'].quantile(.25):.0f} to {ss['n_msgs'].quantile(.75):.0f}",
                "median_duration_min": ss["duration_min"].median(),
                "iqr_duration_min": f"{ss['duration_min'].quantile(.25):.0f} to {ss['duration_min'].quantile(.75):.0f}",
                "median_participants": ss["n_senders"].median(),
                "mean_participants": ss["n_senders"].mean(),
                "n_multi_sender": int((ss["n_senders"] >= 2).sum()),
                "share_multi_sender": float((ss["n_senders"] >= 2).mean()),
                "share_single_message": float((ss["n_msgs"] == 1).mean()),
                "n_within_session_transitions": dl.get("n", 0),
                "within_median_min": dl.get("median"), "within_q1": dl.get("q1"),
                "within_q3": dl.get("q3"), "within_mean": dl.get("mean"),
            })
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "03_sessions_summary.csv", index=False)
    print(summ.round(2).to_string(index=False))

    # multi-sender sessions by initiator (coded IDs), default threshold
    d = sessionise(df, DEFAULT)
    s = session_table(d)
    init = (s.assign(multi=s.n_senders >= 2).groupby(["phase", "initiator"])["multi"]
             .agg(n_sessions="size", share_multi="mean").reset_index())
    init.to_csv(OUT / "03_sessions_by_initiator.csv", index=False)
    print(init.round(2).to_string(index=False))

    # ---- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.hist(x, bins=np.arange(0, 4.3, 0.1), density=True, color="0.85", edgecolor="white")
    ax.plot(xs, ys, color="0.2", lw=1.2, label="KDE")
    for j, lab in enumerate(["burst", "conversational", "idle"]):
        ax.plot(xs, w[j] * norm.pdf(xs, mu[j], sd[j]), lw=1, ls="--", label=f"{lab} ({10 ** mu[j]:.0f} min)")
    for T in THRESHOLDS:
        ax.axvline(np.log10(T), color="#d1495b" if T == DEFAULT else "#999", lw=1)
    ticks = [1, 5, 15, 60, 240, 1440, 10080]
    ax.set_xticks(np.log10(ticks), ["1 m", "5 m", "15 m", "1 h", "4 h", "1 d", "7 d"])
    ax.set_xlabel("Gap between consecutive messages (log scale; zero gaps excluded)")
    ax.set_ylabel("Density")
    ax.set_title(f"A. Inter-message gaps (n={len(x)}); lines at 30, 60, 120 min", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    ax.plot(curve["threshold_min"], curve["conversational_component_retained"], marker="o", ms=3,
            label="conversational gaps kept within session")
    ax.plot(curve["threshold_min"], curve["idle_component_wrongly_retained"], marker="o", ms=3,
            label="idle gaps wrongly kept within session")
    ax2 = ax.twinx()
    ax2.plot(curve["threshold_min"], curve["n_sessions"], color="0.5", lw=1, ls=":", label="sessions (n)")
    ax2.set_ylabel("Sessions (n)", color="0.4")
    ax.set_xscale("log")
    ax.set_xlabel("Session gap threshold (min, log scale)")
    ax.set_ylabel("Proportion of mixture component")
    for T in THRESHOLDS:
        ax.axvline(T, color="#d1495b" if T == DEFAULT else "#999", lw=1)
    ax.legend(fontsize=7, frameon=False, loc="center left")
    ax.set_title("B. Threshold trade-off (3-component mixture)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_gap_distribution.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
