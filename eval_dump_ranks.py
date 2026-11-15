#!/usr/bin/env python3
"""Held-out eval that STORES ALL THE DATA for arbitrary ranking metrics later.

A non-destructive superset of eval_cui_regression.py: same model/inference path,
but per held-out disease it also records

  * recall@TP        — R-precision (K = that disease's own positive count),
  * positive_ranks   — the 1-based ranks of every true gene (a compact, complete
                       record from which recall@K for ANY K, R-precision, and
                       Recall@(n x TP) for ANY n are derivable offline), and
  * all_gene_scores.tsv.gz — the full score of EVERY gene for EVERY held-out
                       disease (disease, gene_id, score), the rawest artifact so
                       any future metric (calibration, negatives, MRR, ...) needs
                       no model re-run.

Outputs use a distinct prefix (default 'cui_rank_dump') so it never overwrites the
original cui_regression_eval_* files.

    python eval_dump_ranks.py <exp_dir> <test_tsv> [--out-prefix cui_rank_dump] [--no-scores]
"""
import argparse
import gzip
import os.path as osp

import numpy as np
import pandas as pd
import torch
import yaml
from box import Box

from source.DisGeneFormer import DisGeneFormer
from source.GeneNet import GeneNet
from source.DiseaseNet import DiseaseNet


def spearman(pred, true):
    if len(pred) < 2:
        return float("nan")
    pr = pd.Series(pred).rank().to_numpy()
    tr = pd.Series(true).rank().to_numpy()
    if pr.std() == 0 or tr.std() == 0:
        return float("nan")
    return float(np.corrcoef(pr, tr)[0, 1])


def pearson(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def auc_from_scores(pos, neg):
    """Rank-based AUC: P(score(pos) > score(neg))."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = len(pos), len(neg)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("test_tsv")
    ap.add_argument("--neg-ratio", type=float, default=1.0,
                    help="sampled negatives per positive, per held-out disease")
    ap.add_argument("--top-k", default="5,20,50")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="cui_rank_dump",
                    help="output basename prefix (kept distinct so originals are preserved)")
    ap.add_argument("--no-scores", action="store_true",
                    help="skip the full all_gene_scores.tsv.gz dump (still writes ranks/recall@TP)")
    a = ap.parse_args()
    exp = a.exp_dir
    rng = np.random.default_rng(a.seed)
    ks = [int(x) for x in a.top_k.split(",")]
    dump_scores = not a.no_scores

    with open(osp.join(exp, "config.yml")) as f:
        cfg = Box(yaml.safe_load(f))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dump] device={device}  exp={exp}  dump_scores={dump_scores}")

    gene_net = GeneNet(cfg=cfg, experiment_dir=exp)[0].to(device)
    disease_net = DiseaseNet(cfg=cfg, experiment_dir=exp)[0].to(device)

    dg = pd.read_csv(osp.join(exp, "gene_net/processed/gene_id_index_mapping.tsv"),
                     sep="\t", header=None, skiprows=1)
    gene_id_to_idx = dict(zip(dg[0].astype(int), dg[1].astype(int)))
    dd = pd.read_csv(osp.join(exp, "disease_net/processed/disease_id_feature_index_mapping.tsv"),
                     sep="\t", header=None, skiprows=1)
    dis_id_to_idx = dict(zip(dd[0].astype(str), dd[1].astype(int)))

    mcfg = cfg.get("model", {})
    task = mcfg.get("task", "classification")
    ablate = {k: bool(v) for k, v in mcfg.get("ablate", {}).items() if v}
    bip_layers = int(mcfg.get("bipartite_layers", 0))
    _kw = {}
    if bip_layers > 0:
        from source.DisGeneFormer_bip import DisGeneFormer as _DGF
        _kw = dict(bipartite_layers=bip_layers,
                   bipartite_edge_dropout=float(mcfg.get("bipartite_edge_dropout", 0.5)))
    else:
        _DGF = DisGeneFormer
    model = _DGF(
        gene_feature_dim=gene_net.x.shape[1],
        disease_feature_dim=disease_net.x.shape[1],
        fc_hidden_dim=cfg.model.fc_hidden_dim,
        gene_net_hidden_dim=cfg.model.gene_net_hidden_dim,
        disease_net_hidden_dim=cfg.model.disease_net_hidden_dim,
        mode="DGP", ablate=ablate, task=task,
        transformer_num_heads=int(mcfg.get("transformer_num_heads", 4)),
        transformer_num_layers=int(mcfg.get("transformer_num_layers", 2)),
        transformer_dropout=float(mcfg.get("transformer_dropout", 0.1)),
        **_kw,
    ).to(device)
    model_path = osp.join(exp, "best_model.ptm")
    if not osp.isfile(model_path):
        model_path = osp.join(exp, "best_model_fold_1.ptm")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[dump] loaded {osp.basename(model_path)} | task={task} | ablate={ablate}")

    _assoc_dis = set()
    if bip_layers > 0:
        train_pos_path = cfg.get("data", {}).get("train_positives_path",
                                                 "data/training/train_disgenet_cui_train.tsv")
        tp = pd.read_csv(train_pos_path, sep="\t", header=None, names=["g", "d", "s"])
        tp = tp[tp["g"].astype(int).isin(gene_id_to_idx) & tp["d"].astype(str).isin(dis_id_to_idx)]
        _ag = np.array([gene_id_to_idx[int(x)] for x in tp["g"]], dtype=np.int64)
        _ad = np.array([dis_id_to_idx[str(x)] for x in tp["d"]], dtype=np.int64)
        model.set_assoc_edges(_ag, _ad)
        _assoc_dis = set(tp["d"].astype(str))
        print(f"[bipartite] eval assoc_edges={len(_ag)} from {train_pos_path} (train-split only)")

    with torch.no_grad():
        g_emb, d_emb = model.encode_nodes(gene_net, disease_net)
    head = model.final_fc.to(device).eval()

    def predict(pair_idx):
        out = np.empty(len(pair_idx), dtype=np.float32)
        gi = torch.as_tensor(pair_idx[:, 0], device=device, dtype=torch.long)
        di = torch.as_tensor(pair_idx[:, 1], device=device, dtype=torch.long)
        batch = 500_000
        with torch.no_grad():
            for i in range(0, len(pair_idx), batch):
                feats = torch.cat([g_emb[gi[i:i + batch]], d_emb[di[i:i + batch]]], dim=1)
                logits = head(feats).float()
                vals = torch.sigmoid(logits[:, 0]) if task == "regression" else logits[:, 1]
                out[i:i + batch] = vals.cpu().numpy()
        return out

    # ---- held-out positives ----
    test = pd.read_csv(a.test_tsv, sep="\t", header=None, names=["gene", "dis", "score"])
    test["gene"] = test["gene"].astype(int)
    test["dis"] = test["dis"].astype(str)
    before = len(test)
    test = test[test["gene"].isin(gene_id_to_idx) & test["dis"].isin(dis_id_to_idx)]
    print(f"[dump] test positives: {len(test)}/{before} usable "
          f"({test['dis'].nunique()} diseases, {test['gene'].nunique()} genes)")
    if bip_layers > 0:
        leak = _assoc_dis & set(test["dis"].astype(str))
        assert not leak, f"[bipartite] LEAKAGE: {len(leak)} test diseases appear in association edges"
    if len(test) == 0:
        raise SystemExit("[dump] no usable test pairs — ids not in the trained graph mappings.")

    pos_pairs = np.array([[gene_id_to_idx[g], dis_id_to_idx[d]]
                          for g, d in zip(test["gene"], test["dis"])], dtype=np.int64)
    pos_true = test["score"].to_numpy(dtype=np.float32)
    pos_pred = predict(pos_pairs)

    # ---- sampled negatives per held-out disease (for AUC parity with original) ----
    all_gene_ids = np.array(sorted(gene_id_to_idx), dtype=np.int64)
    pos_by_dis = test.groupby("dis")["gene"].apply(set).to_dict()
    neg_rows = []
    for d, pg in pos_by_dis.items():
        k = max(1, int(round(len(pg) * a.neg_ratio)))
        cand = np.setdiff1d(all_gene_ids, np.array(sorted(pg), dtype=np.int64))
        k = min(k, len(cand))
        for g in rng.choice(cand, size=k, replace=False):
            neg_rows.append([gene_id_to_idx[int(g)], dis_id_to_idx[d]])
    neg_pairs = np.array(neg_rows, dtype=np.int64)
    neg_pred = predict(neg_pairs)

    mse = float(np.mean((pos_pred - pos_true) ** 2))
    mae = float(np.mean(np.abs(pos_pred - pos_true)))
    ss_res = float(np.sum((pos_true - pos_pred) ** 2))
    ss_tot = float(np.sum((pos_true - pos_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    auc = auc_from_scores(pos_pred, neg_pred)

    # ---- per-disease: full ranking, recall@k, recall@TP, positive_ranks, score dump ----
    gene_idx_all = np.array([gene_id_to_idx[int(g)] for g in all_gene_ids], dtype=np.int64)
    per_disease = []
    recalls = {k: [] for k in ks}
    rprec = []  # recall@TP (R-precision)

    scores_path = osp.join(exp, "all_gene_scores.tsv.gz")
    sfh = gzip.open(scores_path, "wt") if dump_scores else None
    if sfh:
        sfh.write("disease\tgene_id\tscore\n")

    for d, pg in pos_by_dis.items():
        di = dis_id_to_idx[d]
        pairs = np.column_stack([gene_idx_all, np.full(len(gene_idx_all), di, dtype=np.int64)])
        sc = predict(pairs)
        order = np.argsort(-sc)
        ranked = all_gene_ids[order]
        pg_arr = np.array(sorted(int(x) for x in pg), dtype=np.int64)
        tp = len(pg_arr)
        # 1-based ranks of this disease's positive genes, ascending
        pos_ranks = np.sort(np.flatnonzero(np.isin(ranked, pg_arr)) + 1)

        row = {"disease": d, "n_positives": tp}
        for k in ks:
            rec = int(np.sum(pos_ranks <= k)) / tp
            recalls[k].append(rec)
            row[f"recall@{k}"] = rec
        rtp = int(np.sum(pos_ranks <= tp)) / tp
        rprec.append(rtp)
        row["recall@TP"] = rtp
        row["positive_ranks"] = ";".join(map(str, pos_ranks.tolist()))
        per_disease.append(row)

        if sfh:
            # store every gene's score for this disease (rawest artifact)
            sfh.write("".join(f"{d}\t{int(g)}\t{s:.6g}\n"
                              for g, s in zip(all_gene_ids, sc)))
    if sfh:
        sfh.close()

    per_csv = osp.join(exp, f"{a.out_prefix}_per_disease.csv")
    pd.DataFrame(per_disease).to_csv(per_csv, index=False)

    lines = [
        "=== CUI held-out eval + rank dump ===",
        f"model            : {osp.basename(model_path)}  (task={task}, ablate={ablate})",
        f"test positives   : {len(pos_pred)}   negatives: {len(neg_pred)}",
        f"test diseases    : {len(pos_by_dis)}   genes: {len(all_gene_ids)}",
        "--- positive vs. sampled-negative separation ---",
        f"AUC              : {auc:.5f}",
        "--- score regression (sanity vs original) ---",
        f"MSE {mse:.5f}  MAE {mae:.5f}  R2 {r2:.5f}  "
        f"Pearson {pearson(pos_pred, pos_true):.5f}  Spearman {spearman(pos_pred, pos_true):.5f}",
        "--- ranking (mean over held-out diseases) ---",
    ] + [f"recall@{k:<9}: {np.mean(recalls[k]):.4f}" for k in ks] + [
        f"recall@TP (R-prec): {np.mean(rprec):.4f}",
        "",
        f"per-disease (+ recall@TP, positive_ranks): {osp.basename(per_csv)}",
        f"full gene scores            : {osp.basename(scores_path) if dump_scores else '(skipped)'}",
    ]
    summary = "\n".join(lines)
    with open(osp.join(exp, f"{a.out_prefix}_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
