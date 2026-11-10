#!/usr/bin/env python3
"""Held-out regression evaluation for a CUI-space DisGeneFormer model.

Loads the trained model + graphs from <exp_dir>, scores a held-out
(gene, CUI, score) test set (positives) plus randomly-sampled negatives, and
reports regression metrics on the predicted score (MSE/MAE/R2/Pearson/Spearman),
positive-vs-negative separation (AUC), and top-K recall per held-out disease
(rank all genes for a disease, how many held-out positives land in the top K).

This replaces the OMIM-centric evaluate_fold.py for CUI-keyed data: it reuses the
exact inference path from predict_genes_fold.py (encode_nodes -> final_fc ->
sigmoid) but on an arbitrary held-out pair list instead of OMIM ground truth.

    python eval_cui_regression.py <exp_dir> <test_tsv>

Writes <exp_dir>/cui_regression_eval_summary.txt and
       <exp_dir>/cui_regression_eval_per_disease.csv
"""
import argparse
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
    a = ap.parse_args()
    exp = a.exp_dir
    rng = np.random.default_rng(a.seed)
    ks = [int(x) for x in a.top_k.split(",")]

    with open(osp.join(exp, "config.yml")) as f:
        cfg = Box(yaml.safe_load(f))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}  exp={exp}")

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
    model = DisGeneFormer(
        gene_feature_dim=gene_net.x.shape[1],
        disease_feature_dim=disease_net.x.shape[1],
        fc_hidden_dim=cfg.model.fc_hidden_dim,
        gene_net_hidden_dim=cfg.model.gene_net_hidden_dim,
        disease_net_hidden_dim=cfg.model.disease_net_hidden_dim,
        mode="DGP", ablate=ablate, task=task,
        transformer_num_heads=int(mcfg.get("transformer_num_heads", 4)),
        transformer_num_layers=int(mcfg.get("transformer_num_layers", 2)),
        transformer_dropout=float(mcfg.get("transformer_dropout", 0.1)),
    ).to(device)
    model_path = osp.join(exp, "best_model.ptm")
    if not osp.isfile(model_path):
        model_path = osp.join(exp, "best_model_fold_1.ptm")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[eval] loaded {osp.basename(model_path)} | task={task} | ablate={ablate}")

    # Cache node embeddings once; the head is cheap to run over many pairs.
    with torch.no_grad():
        g_emb, d_emb = model.encode_nodes(gene_net, disease_net)
    head = model.final_fc.to(device).eval()

    def predict(pair_idx):
        """pair_idx: (N,2) int array of (gene_idx, disease_idx) -> predicted scores."""
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
    print(f"[eval] test positives: {len(test)}/{before} usable "
          f"({test['dis'].nunique()} diseases, {test['gene'].nunique()} genes)")
    if len(test) == 0:
        raise SystemExit("[eval] no usable test pairs — test gene/disease ids are not in "
                         "the trained graph mappings. Check that the disease graph was built "
                         "from the CUI raw files (disease_net.raw_root) and includes held-out diseases.")

    pos_pairs = np.array([[gene_id_to_idx[g], dis_id_to_idx[d]]
                          for g, d in zip(test["gene"], test["dis"])], dtype=np.int64)
    pos_true = test["score"].to_numpy(dtype=np.float32)
    pos_pred = predict(pos_pairs)

    # ---- sampled negatives per held-out disease (random non-positive genes) ----
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

    # ---- regression metrics on the predicted score (positives) ----
    mse = float(np.mean((pos_pred - pos_true) ** 2))
    mae = float(np.mean(np.abs(pos_pred - pos_true)))
    ss_res = float(np.sum((pos_true - pos_pred) ** 2))
    ss_tot = float(np.sum((pos_true - pos_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    auc = auc_from_scores(pos_pred, neg_pred)

    # ---- top-K recall per held-out disease (rank ALL genes) ----
    gene_idx_all = np.array([gene_id_to_idx[int(g)] for g in all_gene_ids], dtype=np.int64)
    per_disease = []
    recalls = {k: [] for k in ks}
    for d, pg in pos_by_dis.items():
        di = dis_id_to_idx[d]
        pairs = np.column_stack([gene_idx_all, np.full(len(gene_idx_all), di, dtype=np.int64)])
        sc = predict(pairs)
        ranked = all_gene_ids[np.argsort(-sc)]
        pg_set = set(int(x) for x in pg)
        row = {"disease": d, "n_positives": len(pg_set)}
        for k in ks:
            hit = len(set(int(x) for x in ranked[:k]) & pg_set)
            rec = hit / len(pg_set)
            recalls[k].append(rec)
            row[f"recall@{k}"] = rec
        per_disease.append(row)

    pd.DataFrame(per_disease).to_csv(
        osp.join(exp, "cui_regression_eval_per_disease.csv"), index=False)

    lines = [
        "=== CUI held-out regression eval ===",
        f"model            : {osp.basename(model_path)}  (task={task}, ablate={ablate})",
        f"test positives   : {len(pos_pred)}   negatives: {len(neg_pred)}",
        f"test diseases    : {len(pos_by_dis)}",
        "--- score regression (positives: predicted vs true DisGeNET score) ---",
        f"MSE              : {mse:.5f}",
        f"MAE              : {mae:.5f}",
        f"R2               : {r2:.5f}",
        f"Pearson r        : {pearson(pos_pred, pos_true):.5f}",
        f"Spearman rho     : {spearman(pos_pred, pos_true):.5f}",
        "--- positive vs. sampled-negative separation ---",
        f"AUC              : {auc:.5f}",
        "--- ranking (mean over held-out diseases) ---",
    ] + [f"recall@{k:<10}: {np.mean(recalls[k]):.4f}" for k in ks]
    summary = "\n".join(lines)
    with open(osp.join(exp, "cui_regression_eval_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
