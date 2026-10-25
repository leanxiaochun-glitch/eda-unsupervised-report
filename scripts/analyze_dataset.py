#!/usr/bin/env python3
"""Generic, label-isolated EDA + unsupervised analysis for tabular data.

Writes only aggregate metrics, summaries and figures; never exports row-level data.
"""
from __future__ import annotations
import argparse, json, math, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
BLUE, RED, GRAY, GREEN, ORANGE = "#2E6FB7", "#C0392B", "#999999", "#4C9F70", "#E67E22"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset-name", default="数据集")
    p.add_argument("--sheet", default=0, help="XLSX sheet name or zero-based index")
    p.add_argument("--label", default=None)
    p.add_argument("--positive-label", default="1")
    p.add_argument("--id-columns", default="", help="Comma-separated columns excluded from features")
    p.add_argument("--max-missing", type=float, default=0.80)
    p.add_argument("--min-row-completeness", type=float, default=0.50)
    p.add_argument("--max-features", type=int, default=120)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=3000, help="Maximum rows used for t-SNE only")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_table(path, sheet):
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        try:
            sheet = int(sheet)
        except (ValueError, TypeError):
            pass
        return pd.read_excel(path, sheet_name=sheet)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    for enc in ["utf-8-sig", "utf-8", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unsupported or unreadable input: {path}")


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def robust_auc(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.nan
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(u / (len(a) * len(b)))


def md_table(rows, columns):
    if not rows:
        return "（无可报告结果）"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()
    src, out = Path(args.input).expanduser().resolve(), Path(args.output).expanduser().resolve()
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    raw = read_table(src, args.sheet)
    raw = raw.dropna(axis=1, how="all").copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    if len(set(raw.columns)) != len(raw.columns):
        raise ValueError("Duplicate column names found; rename them before analysis.")
    n_raw, p_raw = raw.shape
    explicit_exclude = {c.strip() for c in args.id_columns.split(",") if c.strip()}
    if args.label and args.label not in raw.columns:
        raise ValueError(f"Label column not found: {args.label}")
    y = None
    positive = args.positive_label
    if args.label:
        label_raw = raw[args.label]
        y = (label_raw.astype(str).str.strip() == str(positive).strip()).astype(float)
        y[label_raw.isna()] = np.nan
    exclude = set(explicit_exclude)
    if args.label:
        exclude.add(args.label)
    id_pattern = re.compile(r"(^|_)(id|code|uuid|guid|key|date|time|timestamp|日期|时间|代码|编号)($|_)", re.I)
    auto_exclude = []
    numeric = {}
    conversion = {}
    for c in raw.columns:
        s = raw[c]
        if c in exclude or pd.api.types.is_datetime64_any_dtype(s):
            continue
        if id_pattern.search(c) or (s.nunique(dropna=True) / max(len(s), 1) > 0.98 and len(s) > 20 and not pd.api.types.is_numeric_dtype(s)):
            auto_exclude.append(c)
            continue
        if pd.api.types.is_numeric_dtype(s):
            z = pd.to_numeric(s, errors="coerce")
            convertible = 1.0
        else:
            z = pd.to_numeric(s, errors="coerce")
            convertible = float(z.notna().sum() / max(s.notna().sum(), 1))
        conversion[c] = convertible
        if convertible >= 0.80 and z.notna().sum() >= 2:
            numeric[c] = z.replace([np.inf, -np.inf], np.nan)
    if not numeric:
        raise ValueError("No usable numeric features. Supply numeric data or an appropriate encoding plan.")
    X0 = pd.DataFrame(numeric)
    col_missing = X0.isna().mean()
    dropped_missing = col_missing[col_missing > args.max_missing].index.tolist()
    X0 = X0.drop(columns=dropped_missing)
    dropped_constant = X0.columns[X0.nunique(dropna=True) <= 1].tolist()
    X0 = X0.drop(columns=dropped_constant)
    if X0.shape[1] == 0:
        raise ValueError("All candidate features were removed; relax --max-missing or inspect the data.")
    row_completeness = X0.notna().mean(axis=1)
    keep = row_completeness >= args.min_row_completeness
    X0 = X0.loc[keep].copy()
    y = y.loc[keep] if y is not None else None
    if len(X0) < 8:
        raise ValueError("Too few complete rows; lower --min-row-completeness or inspect block missingness.")
    # Missing -> median -> winsorize -> robust scale. No label is used here.
    med = X0.median()
    Xf = X0.fillna(med)
    lo, hi = Xf.quantile(0.01), Xf.quantile(0.99)
    Xw = Xf.clip(lo, hi, axis=1)
    q1, q3 = Xw.quantile(0.25), Xw.quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    Xs_all = ((Xw - Xw.median()) / iqr).dropna(axis=1, how="all").fillna(0.0)
    dropped_zero_iqr = [c for c in Xw.columns if c not in Xs_all.columns]
    Xw = Xw[Xs_all.columns]
    if Xs_all.shape[1] > args.max_features:
        selected = Xs_all.var().sort_values(ascending=False).head(args.max_features).index
        Xs_all, Xw = Xs_all[selected], Xw[selected]
    Xs = Xs_all.clip(-10, 10)
    n, p = Xs.shape
    known = y.notna() if y is not None else pd.Series(False, index=Xs.index)
    npos = int(y[known].sum()) if y is not None else 0
    nknown = int(known.sum())
    base = float(npos / nknown) if nknown else None
    metrics = {
        "dataset_name": args.dataset_name, "source_file": src.name,
        "raw": {"rows": n_raw, "columns": p_raw, "duplicate_rows": int(raw.duplicated().sum())},
        "analysis": {"rows": n, "features": p, "excluded_rows": int(n_raw - n),
                     "max_missing": args.max_missing, "min_row_completeness": args.min_row_completeness},
        "label": None, "quality": {"auto_excluded": auto_exclude, "explicit_excluded": sorted(explicit_exclude),
                                   "dropped_high_missing": dropped_missing, "dropped_constant": dropped_constant,
                                   "dropped_zero_iqr": dropped_zero_iqr,
                                   "high_missing_columns": col_missing[col_missing > 0.5].sort_values(ascending=False).head(20).to_dict(),
                                   "inf_cells": int(sum(np.isinf(pd.to_numeric(raw[c], errors="coerce")).sum() for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])))},
        "features": list(Xs.columns),
    }
    if y is not None:
        metrics["label"] = {"column": args.label, "positive_label": positive, "known": nknown,
                            "positive": npos, "base_rate": base,
                            "imbalance_ratio": (nknown - npos) / npos if npos else None}
    # Univariate analysis
    uni = []
    for c in Xw.columns:
        rec = {"feature": c, "median": float(Xw[c].median()), "iqr": float(Xw[c].quantile(.75) - Xw[c].quantile(.25))}
        if y is not None and npos >= 1 and nknown - npos >= 1:
            a, b = Xw.loc[y == 1, c], Xw.loc[y == 0, c]
            auc = robust_auc(a, b)
            rec.update({"median_positive": float(a.median()), "median_negative": float(b.median()),
                        "auc": auc, "auc_strength": max(auc, 1 - auc),
                        "direction": "positive_higher" if auc >= .5 else "positive_lower"})
        uni.append(rec)
    if y is not None and uni and "auc_strength" in uni[0]:
        uni = sorted(uni, key=lambda r: r["auc_strength"], reverse=True)
    else:
        uni = sorted(uni, key=lambda r: r["iqr"], reverse=True)
    metrics["univariate"] = uni
    # Select largely non-redundant features for plots/interactions without label-based feature selection for clustering.
    top_candidates = [r["feature"] for r in uni[:30]]
    chosen = []
    for c in top_candidates:
        if all(abs(Xw[c].corr(Xw[d], method="spearman")) < .95 for d in chosen):
            chosen.append(c)
        if len(chosen) >= 8:
            break
    corr_cols = chosen[:12] if len(chosen) >= 2 else list(Xw.columns[:min(12, p)])
    corr = Xw[corr_cols].corr(method="spearman")
    strong_pairs = []
    for i, a in enumerate(corr_cols):
        for b in corr_cols[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= .7:
                strong_pairs.append({"a": a, "b": b, "spearman": float(r)})
    metrics["correlation"] = {"columns": corr_cols, "strong_pairs": strong_pairs}
    # Figures 1 and 2
    plot_feats = (chosen or corr_cols)[:4]
    fig, axes = plt.subplots(1, len(plot_feats), figsize=(3.3 * len(plot_feats), 3.4), squeeze=False)
    for ax, c in zip(axes[0], plot_feats):
        if y is not None and npos >= 2:
            data = [Xw.loc[y == 0, c], Xw.loc[y == 1, c]]
            vp = ax.violinplot(data, showmedians=True)
            for i, body in enumerate(vp["bodies"]):
                body.set_facecolor([BLUE, RED][i]); body.set_alpha(.7)
            ax.set_xticks([1, 2]); ax.set_xticklabels(["阴性", "阳性"])
        else:
            ax.hist(Xw[c], bins=30, color=BLUE, alpha=.75)
            ax.set_xticks([])
        ax.set_title(c[:22], fontsize=9)
    fig.suptitle("关键指标分布", fontsize=12); fig.tight_layout()
    savefig(fig, figdir / "fig1_distributions.png")
    if len(corr_cols) >= 2:
        fig, ax = plt.subplots(figsize=(max(6, .62 * len(corr_cols)), max(5, .55 * len(corr_cols))))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_cols))); ax.set_xticklabels(corr_cols, rotation=50, ha="right", fontsize=7)
        ax.set_yticks(range(len(corr_cols))); ax.set_yticklabels(corr_cols, fontsize=7)
        fig.colorbar(im, shrink=.8); ax.set_title("Spearman 相关结构")
        savefig(fig, figdir / "fig2_correlation.png")
    # Interactions and threshold concentrations
    interactions, thresholds = [], []
    if y is not None and npos >= 1 and nknown - npos >= 1 and base > 0:
        pair_pool = chosen[:6] if len(chosen) >= 4 else corr_cols[:6]
        pairs = []
        for i in range(len(pair_pool)):
            for j in range(i + 1, len(pair_pool)):
                pairs.append((pair_pool[i], pair_pool[j]))
        for a, b in pairs[:4]:
            ca, cb = Xw[a].median(), Xw[b].median()
            for av in [False, True]:
                for bv in [False, True]:
                    mask = ((Xw[a] > ca) == av) & ((Xw[b] > cb) == bv)
                    cnt = int(mask.sum()); pos = int(y[mask].sum())
                    rate = pos / cnt if cnt else np.nan
                    interactions.append({"a": a, "b": b, "a_cutoff": float(ca), "b_cutoff": float(cb),
                                         "a_above": av, "b_above": bv, "n": cnt, "positive": pos,
                                         "rate": rate, "concentration": rate / base if cnt else None})
        for c in chosen[:6]:
            cut = Xw[c].median()
            for above in [False, True]:
                mask = (Xw[c] > cut) == above
                cnt = int(mask.sum()); pos = int(y[mask].sum())
                thresholds.append({"feature": c, "cutoff": float(cut), "above": above, "n": cnt,
                                   "positive": pos, "rate": pos / cnt if cnt else None,
                                   "concentration": (pos / cnt / base) if cnt else None})
        metrics["interactions"], metrics["thresholds"] = interactions, thresholds
        if interactions:
            fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
            for ax, (a, b) in zip(axes.flat, pairs[:4]):
                sub = [r for r in interactions if r["a"] == a and r["b"] == b]
                labels = [f"{a[:9]}{'>' if r['a_above'] else '<='}\n{b[:9]}{'>' if r['b_above'] else '<='}" for r in sub]
                vals = [r["concentration"] or 0 for r in sub]
                ax.bar(range(4), vals, color=[RED if v >= 2 else BLUE if v < 1 else GRAY for v in vals])
                ax.axhline(1, color="black", ls="--", lw=1)
                ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7)
                ax.set_title(f"{a[:18]} × {b[:18]}", fontsize=9); ax.set_ylabel("浓度倍数")
                ax.set_ylim(0, max(vals + [1]) * 1.3)
            fig.suptitle("交互组合的标签浓度（事后叠加）"); fig.tight_layout()
            savefig(fig, figdir / "fig3_interactions.png")
    # PCA
    pca = PCA().fit(Xs)
    evr, cum = pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)
    Z = pca.transform(Xs)
    n80 = int(np.argmax(cum >= .8)) + 1 if np.any(cum >= .8) else p
    n90 = int(np.argmax(cum >= .9)) + 1 if np.any(cum >= .9) else p
    loadings = []
    for i in range(min(3, p)):
        s = pd.Series(pca.components_[i], index=Xs.columns).sort_values(key=abs, ascending=False)
        loadings.append({"component": i + 1, "variance_ratio": float(evr[i]),
                         "top_features": [{"feature": c, "loading": float(v)} for c, v in s.head(8).items()]})
    metrics["pca"] = {"explained_first3": evr[:3].tolist(), "cumulative_first10": float(cum[min(9, p - 1)]),
                      "n_components_80": n80, "n_components_90": n90, "loadings": loadings}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    m = min(30, p)
    axes[0].plot(range(1, m + 1), cum[:m] * 100, "o-", color=BLUE, ms=4)
    axes[0].axhline(80, color=RED, ls="--", lw=1); axes[0].set_title("PCA 累计方差解释曲线")
    axes[0].set_xlabel("主成分个数"); axes[0].set_ylabel("累计解释率(%)")
    axes[1].scatter(Z[:, 0], Z[:, 1], s=7, c=GRAY, alpha=.35)
    if y is not None and npos:
        axes[1].scatter(Z[y == 1, 0], Z[y == 1, 1], s=70, facecolors="none", edgecolors=RED, label="阳性（事后叠加）")
        axes[1].legend()
    axes[1].set_title("PC1 × PC2"); axes[1].set_xlabel(f"PC1 ({evr[0]:.1%})"); axes[1].set_ylabel(f"PC2 ({evr[1]:.1%})")
    savefig(fig, figdir / "fig4_pca.png")
    # t-SNE on deterministic subsample only
    rng = np.random.default_rng(args.seed)
    tsne_idx = np.arange(n) if n <= args.max_rows else np.sort(rng.choice(n, args.max_rows, replace=False))
    Xt = Xs.iloc[tsne_idx]
    perplexity = max(2, min(30, (len(Xt) - 1) // 3))
    T = TSNE(n_components=2, perplexity=perplexity, random_state=args.seed, init="pca", learning_rate="auto").fit_transform(Xt)
    metrics["tsne"] = {"rows": len(Xt), "perplexity": perplexity, "note": "Visualization only; clustering uses original feature space."}
    # K selection and final clustering
    kmax = max(2, min(8, n - 1))
    k_records = []
    for k in range(2, kmax + 1):
        model = KMeans(n_clusters=k, n_init=15, random_state=args.seed).fit(Xs)
        sil = silhouette_score(Xs, model.labels_, sample_size=min(5000, n), random_state=args.seed) if n > k else np.nan
        k_records.append({"k": k, "inertia": float(model.inertia_), "silhouette": float(sil)})
    final_k = min(max(2, args.k), n - 1)
    km = KMeans(n_clusters=final_k, n_init=30, random_state=args.seed).fit(Xs)
    labels = pd.Series(km.labels_, index=Xs.index)
    ward = AgglomerativeClustering(n_clusters=final_k, linkage="ward").fit(Xs)
    ari = adjusted_rand_score(labels, ward.labels_)
    profiles = []
    zmat = pd.DataFrame({c: Xs[labels == c].mean() for c in range(final_k)})
    for c in range(final_k):
        mask = labels == c
        topz = zmat[c].sort_values(key=abs, ascending=False).head(8)
        rec = {"cluster": c, "n": int(mask.sum()), "share": float(mask.mean()),
               "top_z": [{"feature": f, "z": float(v)} for f, v in topz.items()]}
        if y is not None and npos:
            pos = int(y[mask].sum())
            rec.update({"positive": pos, "rate": float(y[mask].mean()),
                        "concentration": float(y[mask].mean() / base) if base else None})
        profiles.append(rec)
    metrics["clustering"] = {"k_selection": k_records, "final_k": final_k,
                             "silhouette_best_k": max(k_records, key=lambda r: r["silhouette"])["k"],
                             "kmeans_ward_ari": float(ari), "profiles": profiles}
    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    kk = pd.DataFrame(k_records)
    ax1.plot(kk.k, kk.inertia / 1000, "o-", color=BLUE); ax1.set_ylabel("Inertia（千）", color=BLUE); ax1.set_xlabel("k")
    ax2 = ax1.twinx(); ax2.plot(kk.k, kk.silhouette, "s--", color=RED); ax2.set_ylabel("轮廓系数", color=RED)
    ax2.axvline(final_k, color=GRAY, ls=":"); ax1.set_title("k 的选择")
    savefig(fig, figdir / "fig6_k_selection.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    palette = [GREEN, RED, "#9BB7D4", ORANGE, "#8E6C8A", "#6C8EBF", "#B0855B", "#555555"]
    for c in range(final_k):
        loc = labels.iloc[tsne_idx].to_numpy() == c
        axes[0].scatter(T[loc, 0], T[loc, 1], s=6, color=palette[c % len(palette)], alpha=.5, label=f"簇{c}")
    axes[0].set_title("t-SNE · 无监督簇结构"); axes[0].legend(markerscale=2, fontsize=7)
    axes[1].scatter(T[:, 0], T[:, 1], s=6, c=GRAY, alpha=.3)
    if y is not None and npos:
        pos_t = y.iloc[tsne_idx].to_numpy() == 1
        axes[1].scatter(T[pos_t, 0], T[pos_t, 1], s=80, facecolors="none", edgecolors=RED, label="阳性")
        axes[1].legend()
    axes[1].set_title("t-SNE · 标签事后叠加")
    savefig(fig, figdir / "fig5_tsne.png")
    topz_features = zmat.abs().max(axis=1).sort_values(ascending=False).head(min(20, p)).index
    zshow = zmat.loc[topz_features]
    fig, ax = plt.subplots(figsize=(7.8, max(4, .32 * len(zshow))))
    im = ax.imshow(zshow, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto")
    ax.set_xticks(range(final_k)); ax.set_xticklabels([f"簇{c}" for c in zshow.columns])
    ax.set_yticks(range(len(zshow))); ax.set_yticklabels(zshow.index, fontsize=7)
    fig.colorbar(im, shrink=.8); ax.set_title("簇内均值 z 值画像")
    savefig(fig, figdir / "fig7_cluster_z.png")
    fig, ax1 = plt.subplots(figsize=(8, 4.3))
    sizes = [r["share"] * 100 for r in profiles]
    ax1.bar(np.arange(final_k) - .18, sizes, width=.36, color=BLUE); ax1.set_ylabel("样本占比(%)", color=BLUE)
    ax1.set_xticks(range(final_k)); ax1.set_xticklabels([f"簇{c}" for c in range(final_k)])
    ax2 = ax1.twinx()
    if y is not None and npos:
        conc = [r["concentration"] or 0 for r in profiles]
        ax2.bar(np.arange(final_k) + .18, conc, width=.36, color=RED); ax2.axhline(1, color="black", ls="--", lw=1)
        ax2.set_ylabel("阳性浓度倍数", color=RED)
    else:
        sils = [float(np.mean(Xs[labels == c].var())) for c in range(final_k)]
        ax2.bar(np.arange(final_k) + .18, sils, width=.36, color=ORANGE); ax2.set_ylabel("簇内平均方差")
    ax1.set_title("各簇规模与结构/事后表现")
    savefig(fig, figdir / "fig8_clusters.png")
    # Post-hoc label validation only
    if y is not None and npos >= 1 and nknown - npos >= 1:
        valid = y.notna()
        ct = pd.crosstab(labels[valid], y[valid])
        chi2, pval, dof, expected = stats.chi2_contingency(ct)
        cramv = math.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))) if min(ct.shape) > 1 else np.nan
        ranked = sorted(profiles, key=lambda r: r.get("rate", -1), reverse=True)
        pools = []
        for count in [1, 2]:
            selected = [r["cluster"] for r in ranked[:count]]
            mask = labels.isin(selected) & valid
            captured = int(y[mask].sum())
            pool_n = int(mask.sum())
            pools.append({"clusters": selected, "n": pool_n, "share": float(pool_n / nknown),
                          "captured_positive": captured, "coverage": captured / npos,
                          "rate": float(y[mask].mean()), "concentration": float(y[mask].mean() / base) if base else None})
        metrics["validation"] = {"contingency": ct.to_dict(), "chi2": float(chi2), "dof": int(dof),
                                 "p_value": float(pval), "cramers_v": float(cramv),
                                 "min_expected": float(expected.min()), "risk_pools": pools,
                                 "small_sample_warning": bool(npos < 30 or expected.min() < 5)}
    # Aggregate markdown summary for agent verification; not the final report.
    lines = [f"# {args.dataset_name} · 分析结果核验摘要", "",
             f"- 原始数据：{n_raw} 行 × {p_raw} 列；分析样本：{n} 行；特征：{p} 个。",
             f"- 剔除：高缺失 {len(dropped_missing)} 列，常数 {len(dropped_constant)} 列，自动排除 ID/日期/近唯一列 {len(auto_exclude)} 列。", ""]
    if y is not None:
        lines += [f"- 标签已知 {nknown} 行，阳性 {npos} 行，基线 {base:.2%}。", ""]
    lines += ["## 单变量结果", md_table(uni[:12], list(uni[0].keys())), "",
              "## k 选择", md_table(k_records, ["k", "inertia", "silhouette"]), "",
              f"最终 k={final_k}；KMeans 与 Ward 的 ARI={ari:.3f}。", "",
              "## 簇画像", md_table([{**r, "top_z": "; ".join(f"{z['feature']}={z['z']:.2f}" for z in r['top_z'][:3])} for r in profiles], list(profiles[0].keys())), ""]
    if "validation" in metrics:
        v = metrics["validation"]
        lines += ["## 事后验证", f"卡方 p={v['p_value']:.4g}；Cramér's V={v['cramers_v']:.3f}。",
                  md_table(v["risk_pools"], ["clusters", "n", "share", "captured_positive", "coverage", "rate", "concentration"])]
    lines += ["", "## 图文件", "fig1_distributions.png；fig2_correlation.png；fig3_interactions.png（有标签时）；fig4_pca.png；fig5_tsne.png；fig6_k_selection.png；fig7_cluster_z.png；fig8_clusters.png"]
    (out / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(jsonable(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "rows": n, "features": p, "positive": npos,
                      "final_k": final_k, "ari": ari, "figures": len(list(figdir.glob('*.png')))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
