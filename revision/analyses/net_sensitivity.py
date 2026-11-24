"""R2 #20 network sensitivity analyses (Jheeng archive): windows, sequential ties, isolates,
role proxy, weighted vs unweighted centrality, leave-one-sender-out and day-block bootstrap.

Outputs (data/Jheeng/revision/network_phase/):
  A1_window_sensitivity.csv, A1_direction_check.csv, A2_sequential.csv, A3_isolates.csv,
  A5_role_proxy_ties.csv, A6_node_centralities.csv, A6_rank_agreement.csv,
  A7_loso.csv, A7_bootstrap.csv, A7_bootstrap_contrasts.csv, A7_bootstrap_replicates.csv,
  A7_gini_variants.csv, fig_A1_window_sensitivity.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from net_common import (OUT, PHASE_NAMES, PHASES, coactivity_edges, flag_case_reports, gini, gini_corrected,
                        graph_from_edges, load_messages, network_summary, node_centralities, sender_frame,
                        sequential_edges)

WINDOWS = [5, 10, 15, 30, 60, 120]
SLICES = PHASE_NAMES + ["Overall"]
RNG = np.random.default_rng(20261111)
N_BOOT = 2000


def slice_msgs(msgs, ph):
    return msgs if ph == "Overall" else msgs[msgs["phase"] == ph]


def a1_windows(msgs):
    rows = []
    for w in WINDOWS:
        for ph in SLICES:
            sub = slice_msgs(msgs, ph)
            s = network_summary(sub, coactivity_edges(sub, w))
            G = graph_from_edges(coactivity_edges(sub, w))
            cnt = sub.groupby("sender_code").size()
            s["gini_nodes_only"] = gini(cnt[cnt.index.isin(G.nodes)].values)
            rows.append({"window_min": w, "phase": ph, **s})
    res = pd.DataFrame(rows)
    # direction-of-change check against the 30-minute specification
    metrics = ["nodes", "edges", "density", "mean_partners", "density_incl_isolates",
               "mean_partners_incl_isolates", "gini_nodes_only", "top_betweenness_unw_value",
               "top_betweenness_invw_value"]
    ref = res[res.window_min == 30].set_index("phase")
    dchk = []
    for w in WINDOWS:
        cur = res[res.window_min == w].set_index("phase")
        for m in metrics:
            for a, b in [("2024", "2025"), ("2025", "2026"), ("2024", "2026")]:
                d_ref = np.sign(round(ref.loc[b, m] - ref.loc[a, m], 6))
                d_cur = np.sign(round(cur.loc[b, m] - cur.loc[a, m], 6))
                dchk.append({"window_min": w, "metric": m, "contrast": f"{a}->{b}",
                             "change": round(cur.loc[b, m] - cur.loc[a, m], 3),
                             "direction_30min": int(d_ref), "direction": int(d_cur),
                             "same_direction": d_ref == d_cur})
    dchk = pd.DataFrame(dchk)
    res.to_csv(OUT / "A1_window_sensitivity.csv", index=False)
    dchk.to_csv(OUT / "A1_direction_check.csv", index=False)
    return res, dchk


def a2_sequential(msgs):
    rows = []
    for spec, cap in [("sequential_any_gap", None), ("sequential_le30min", 30)]:
        for ph in SLICES:
            sub = slice_msgs(msgs, ph)
            e = sequential_edges(sub, cap)
            s = network_summary(sub, e)
            rows.append({"specification": spec, "phase": ph, **s})
    res = pd.DataFrame(rows)
    # overlap of sequential edges with 30-min co-activity edges
    ov = []
    for ph in SLICES:
        sub = slice_msgs(msgs, ph)
        ce = {tuple(sorted(x)) for x in coactivity_edges(sub, 30)[["source", "target"]].itertuples(index=False)}
        se = {tuple(sorted(x)) for x in sequential_edges(sub, None)[["source", "target"]].itertuples(index=False)}
        ov.append({"phase": ph, "coactivity_edges": len(ce), "sequential_edges": len(se),
                   "shared": len(ce & se), "jaccard": len(ce & se) / len(ce | se) if ce | se else np.nan})
    res = res.merge(pd.DataFrame(ov), on="phase", how="left")
    res.to_csv(OUT / "A2_sequential.csv", index=False)
    return res


def a3_isolates(msgs):
    rows = []
    for w in WINDOWS:
        for ph in SLICES:
            sub = slice_msgs(msgs, ph)
            G = graph_from_edges(coactivity_edges(sub, w))
            cnt = sub.groupby("sender_code").size()
            for s, n in cnt.items():
                if s not in G.nodes:
                    rows.append({"window_min": w, "phase": ph, "sender_code": s, "messages": int(n)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "A3_isolates.csv", index=False)
    return res


def a5_roles(msgs, df):
    cr_flag = flag_case_reports(df)
    reports = df[cr_flag].groupby("sender_code").size()
    reporters = set(reports[reports >= 5].index)
    rows = []
    for spec in ["coactivity_30min", "sequential_any_gap"]:
        for ph in SLICES:
            sub = slice_msgs(msgs, ph)
            e = coactivity_edges(sub, 30) if spec.startswith("co") else sequential_edges(sub, None)
            if e.empty:
                continue
            cross = e.apply(lambda r: (r.source in reporters) != (r.target in reporters), axis=1)
            G = graph_from_edges(e)
            nr = sum(n in reporters for n in G.nodes)
            no = G.number_of_nodes() - nr
            npairs = G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
            rows.append({"specification": spec, "phase": ph, "reporter_nodes": nr, "other_nodes": no,
                         "edges": len(e), "cross_role_edges": int(cross.sum()),
                         "cross_role_edge_share": cross.mean(),
                         "cross_role_share_if_random": nr * no / npairs if npairs else np.nan,
                         "within_reporter_edges": int(((~cross) & e.source.isin(reporters)).sum()),
                         "within_other_edges": int(((~cross) & ~e.source.isin(reporters)).sum()),
                         "cross_role_weight_share": e.weight[cross].sum() / e.weight.sum()})
    res = pd.DataFrame(rows)
    res.attrs["reporters"] = sorted(reporters)
    res.to_csv(OUT / "A5_role_proxy_ties.csv", index=False)
    return res, sorted(reporters), reports


def a6_centrality(msgs):
    nodes, agree = [], []
    pairs = [("degree", "strength"), ("betweenness_unweighted", "betweenness_inverse_weight"),
             ("betweenness_weight_as_distance_original", "betweenness_inverse_weight"),
             ("eigenvector_unweighted", "eigenvector_weighted"), ("degree", "betweenness_inverse_weight"),
             ("strength", "eigenvector_weighted")]
    for spec in ["coactivity_30min", "sequential_any_gap"]:
        for ph in SLICES:
            sub = slice_msgs(msgs, ph)
            e = coactivity_edges(sub, 30) if spec.startswith("co") else sequential_edges(sub, None)
            nc = node_centralities(graph_from_edges(e))
            nc.insert(0, "phase", ph)
            nc.insert(0, "specification", spec)
            cnt = sub.groupby("sender_code").size()
            nc["messages"] = nc["sender_code"].map(cnt)
            nodes.append(nc)
            for a, b in pairs + [("messages", "strength"), ("messages", "degree")]:
                rho, p = spearmanr(nc[a], nc[b])
                agree.append({"specification": spec, "phase": ph, "x": a, "y": b, "n_nodes": len(nc),
                              "spearman_rho": rho, "p": p,
                              "top_x": nc.sort_values(a, ascending=False).sender_code.iloc[0],
                              "top_y": nc.sort_values(b, ascending=False).sender_code.iloc[0]})
    nodes = pd.concat(nodes, ignore_index=True)
    agree = pd.DataFrame(agree)
    nodes.to_csv(OUT / "A6_node_centralities.csv", index=False)
    agree.to_csv(OUT / "A6_rank_agreement.csv", index=False)
    return nodes, agree


def _partner_sets(sub, w=30):
    d = sub.sort_values(["timestamp", "row"]).reset_index(drop=True)
    ts = d["timestamp"].values.astype("datetime64[ns]")
    s = d["sender_code"].values
    win = np.timedelta64(w, "m")
    out = []
    for i in range(len(d)):
        lo = np.searchsorted(ts, ts[i] - win, side="left")
        hi = np.searchsorted(ts, ts[i] + win, side="right")
        out.append({s[j] for j in range(lo, hi) if j != i and s[j] != s[i]})
    d["partners"] = out
    d["day"] = d["timestamp"].dt.normalize()
    return d


def _net_stats(d):
    cnt = d.groupby("sender_code").size()
    edges = set()
    for snd, ps in zip(d["sender_code"], d["partners"]):
        for o in ps:
            edges.add(tuple(sorted((snd, o))))
    nodes = {x for e in edges for x in e}
    n, m = len(nodes), len(edges)
    return {"gini": gini(cnt.values), "gini_corrected": gini_corrected(cnt.values),
            "density": m / (n * (n - 1) / 2) if n > 1 else np.nan,
            "mean_partners": 2 * m / n if n else np.nan, "nodes": n, "edges": m, "active": len(cnt)}


def a7_stability(msgs, df):
    loso, boot = [], []
    for ph in SLICES:
        sub = slice_msgs(msgs, ph)
        base = network_summary(sub, coactivity_edges(sub, 30))
        for s in sorted(sub["sender_code"].unique()):
            rest = sub[sub["sender_code"] != s]
            r = network_summary(rest, coactivity_edges(rest, 30))
            loso.append({"phase": ph, "dropped": s, "dropped_messages": int((sub.sender_code == s).sum()),
                         "gini": r["gini"], "gini_corrected": r["gini_corrected"], "density": r["density"],
                         "mean_partners": r["mean_partners"], "nodes": r["nodes"], "edges": r["edges"],
                         "gini_full": base["gini"], "density_full": base["density"]})
        # day-block bootstrap: resample calendar days (with replacement); ties are those formed
        # by messages on the sampled days (windows computed on the full phase record)
        d = _partner_sets(sub)
        if ph == "Overall":
            days = pd.date_range("2024-09-22", "2026-04-11", freq="D")
        else:
            days = pd.date_range(PHASES[ph][0].normalize(), PHASES[ph][1].normalize(), freq="D")
        by_day = {k: g for k, g in d.groupby("day")}
        for b in range(N_BOOT):
            pick = RNG.choice(days, size=len(days), replace=True)
            parts = [by_day[k] for k in pick if k in by_day]
            st = _net_stats(pd.concat(parts)) if parts else {}
            boot.append({"phase": ph, "rep": b, **st})
    loso, boot = pd.DataFrame(loso), pd.DataFrame(boot)
    loso.to_csv(OUT / "A7_loso.csv", index=False)
    summ = []
    for ph, g in boot.groupby("phase", sort=False):
        for m in ["gini", "gini_corrected", "density", "mean_partners", "nodes"]:
            summ.append({"phase": ph, "metric": m, "boot_median": g[m].median(),
                         "lo95": g[m].quantile(.025), "hi95": g[m].quantile(.975)})
    summ = pd.DataFrame(summ)
    # probability that the later phase exceeds the earlier (independent bootstraps)
    probs = []
    for m in ["gini", "gini_corrected", "density", "mean_partners"]:
        for a, b in [("2024", "2025"), ("2025", "2026"), ("2024", "2026")]:
            x = boot[boot.phase == a][m].values
            y = boot[boot.phase == b][m].values
            probs.append({"metric": m, "contrast": f"{a}->{b}", "P(later>earlier)": float(np.mean(y > x)),
                          "diff_median": float(np.median(y - x)), "diff_lo95": float(np.quantile(y - x, .025)),
                          "diff_hi95": float(np.quantile(y - x, .975))})
    summ.to_csv(OUT / "A7_bootstrap.csv", index=False)
    pd.DataFrame(probs).to_csv(OUT / "A7_bootstrap_contrasts.csv", index=False)
    boot.to_csv(OUT / "A7_bootstrap_replicates.csv", index=False)
    return loso, summ, pd.DataFrame(probs)


def gini_variants(msgs, df):
    """Gini under alternative message definitions (window-independent)."""
    cr = flag_case_reports(df)
    m2 = msgs.assign(case_report=cr.reindex(msgs.index).values)
    rows = []
    for ph in SLICES:
        sub = slice_msgs(m2, ph)
        variants = {
            "all messages (original)": sub,
            "excluding structured case reports": sub[~sub.case_report],
            "text only (excluding voice notes)": sub[sub.message_type == "text"],
            "text only, excluding case reports": sub[(sub.message_type == "text") & ~sub.case_report],
            "voice notes only": sub[sub.message_type == "voice_note"],
            "excluding top sender": sub[sub.sender_code != sub.sender_code.value_counts().idxmax()],
        }
        for k, v in variants.items():
            c = v.groupby("sender_code").size()
            rows.append({"phase": ph, "variant": k, "senders": len(c), "messages": int(c.sum()),
                         "top_sender": c.idxmax() if len(c) else "", "top_share": c.max() / c.sum() if len(c) else np.nan,
                         "gini": gini(c.values), "gini_corrected": gini_corrected(c.values)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "A7_gini_variants.csv", index=False)
    return res


def figures(res, agree):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    cols = {"2024": "#1b9e77", "2025": "#d95f02", "2026": "#7570b3", "Overall": "#666666"}
    for ax, (m, lab) in zip(axes, [("nodes", "Nodes (non-isolated senders)"), ("density", "Density (isolates excluded)"),
                                   ("mean_partners", "Mean co-activity partners"),
                                   ("top_betweenness_invw_value", "Top betweenness (1/weight)")]):
        for ph in SLICES:
            g = res[res.phase == ph]
            ax.plot(g.window_min, g[m], marker="o", color=cols[ph], label=ph, lw=1.5,
                    ls="--" if ph == "Overall" else "-")
        ax.set_xscale("log")
        ax.set_xticks(WINDOWS)
        ax.set_xticklabels(WINDOWS)
        ax.axvline(30, color="grey", lw=.6, ls=":")
        ax.set_xlabel("Co-activity window (min)")
        ax.set_title(lab, fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_A1_window_sensitivity.png", dpi=200)
    plt.close(fig)


def main():
    df = load_messages()
    msgs = sender_frame(df)
    res, dchk = a1_windows(msgs)
    seq = a2_sequential(msgs)
    iso = a3_isolates(msgs)
    roles, reporters, reports = a5_roles(msgs, df)
    nodes, agree = a6_centrality(msgs)
    gv = gini_variants(msgs, df)
    loso, boot, probs = a7_stability(msgs, df)
    figures(res, agree)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(res[["window_min", "phase", "active_senders", "nodes", "edges", "density", "mean_partners",
               "density_incl_isolates", "mean_partners_incl_isolates", "n_isolates", "gini", "gini_nodes_only",
               "top_betweenness_orig", "top_betweenness_unw", "top_betweenness_invw"]].round(2).to_string())
    print(dchk.groupby(["metric", "contrast"]).same_direction.agg(["sum", "count"]).to_string())
    print(seq.round(2).to_string())
    print(iso[iso.window_min == 30].to_string())
    print("reporters", reporters, reports.to_dict())
    print(roles.round(2).to_string())
    print(agree.round(2).to_string())
    print(gv.round(3).to_string())
    print(loso.round(2).to_string())
    print(boot.round(3).to_string())
    print(probs.round(3).to_string())


if __name__ == "__main__":
    main()
