#!/usr/bin/env python3
"""Predicted-score histograms per split (test / train), in-memory (no 700MB dump).
Loads a trained model (bipartite-aware) and, for each positives file, scores every gene for
every disease and accumulates a 50-bin histogram of all / positive / non-positive scores.

    python score_hist_split.py <exp_dir> test:<test_tsv> train:<train_tsv> ...
Writes <exp_dir>/score_hist_<split>.csv (bin_left,all,pos,neg).
"""
import argparse, os.path as osp
import numpy as np, pandas as pd, torch, yaml
from box import Box

from source.DisGeneFormer import DisGeneFormer
from source.GeneNet import GeneNet
from source.DiseaseNet import DiseaseNet

NB = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("splits", nargs="+", help="each 'name:path.tsv' (e.g. test:data/.../test.tsv)")
    a = ap.parse_args()
    exp = a.exp_dir

    with open(osp.join(exp, "config.yml")) as f:
        cfg = Box(yaml.safe_load(f))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    kw = {}
    if bip_layers > 0:
        from source.DisGeneFormer_bip import DisGeneFormer as DGF
        kw = dict(bipartite_layers=bip_layers,
                  bipartite_edge_dropout=float(mcfg.get("bipartite_edge_dropout", 0.5)))
    else:
        DGF = DisGeneFormer
    model = DGF(
        gene_feature_dim=gene_net.x.shape[1], disease_feature_dim=disease_net.x.shape[1],
        fc_hidden_dim=cfg.model.fc_hidden_dim, gene_net_hidden_dim=cfg.model.gene_net_hidden_dim,
        disease_net_hidden_dim=cfg.model.disease_net_hidden_dim, mode="DGP", ablate=ablate, task=task,
        transformer_num_heads=int(mcfg.get("transformer_num_heads", 4)),
        transformer_num_layers=int(mcfg.get("transformer_num_layers", 2)),
        transformer_dropout=float(mcfg.get("transformer_dropout", 0.1)), **kw,
    ).to(device)
    mp = osp.join(exp, "best_model.ptm")
    if not osp.isfile(mp):
        mp = osp.join(exp, "best_model_fold_1.ptm")
    model.load_state_dict(torch.load(mp, map_location=device))
    model.eval()
    if bip_layers > 0:
        tpp = cfg.get("data", {}).get("train_positives_path", "data/training/train_disgenet_cui_train.tsv")
        tp = pd.read_csv(tpp, sep="\t", header=None, names=["g", "d", "s"])
        tp = tp[tp["g"].astype(int).isin(gene_id_to_idx) & tp["d"].astype(str).isin(dis_id_to_idx)]
        model.set_assoc_edges(np.array([gene_id_to_idx[int(x)] for x in tp["g"]], dtype=np.int64),
                              np.array([dis_id_to_idx[str(x)] for x in tp["d"]], dtype=np.int64))

    with torch.no_grad():
        g_emb, d_emb = model.encode_nodes(gene_net, disease_net)
    head = model.final_fc.to(device).eval()

    def predict(pair_idx):
        out = np.empty(len(pair_idx), dtype=np.float32)
        gi = torch.as_tensor(pair_idx[:, 0], device=device, dtype=torch.long)
        di = torch.as_tensor(pair_idx[:, 1], device=device, dtype=torch.long)
        b = 500_000
        with torch.no_grad():
            for i in range(0, len(pair_idx), b):
                feats = torch.cat([g_emb[gi[i:i+b]], d_emb[di[i:i+b]]], dim=1)
                logits = head(feats).float()
                vals = torch.sigmoid(logits[:, 0]) if task == "regression" else logits[:, 1]
                out[i:i+b] = vals.cpu().numpy()
        return out

    all_gene_ids = np.array(sorted(gene_id_to_idx), dtype=np.int64)
    gene_idx_all = np.array([gene_id_to_idx[int(g)] for g in all_gene_ids], dtype=np.int64)

    for spec in a.splits:
        name, path = spec.split(":", 1)
        test = pd.read_csv(path, sep="\t", header=None, names=["gene", "dis", "score"])
        test["gene"] = test["gene"].astype(int); test["dis"] = test["dis"].astype(str)
        test = test[test["gene"].isin(gene_id_to_idx) & test["dis"].isin(dis_id_to_idx)]
        pos_by_dis = test.groupby("dis")["gene"].apply(set).to_dict()
        Hall = np.zeros(NB); Hpos = np.zeros(NB); Hneg = np.zeros(NB)
        for d, pg in pos_by_dis.items():
            di = dis_id_to_idx[d]
            sc = predict(np.column_stack([gene_idx_all, np.full(len(gene_idx_all), di, dtype=np.int64)]))
            b = np.minimum((sc * NB).astype(int), NB - 1)
            posmask = np.isin(all_gene_ids, np.array(sorted(int(x) for x in pg), dtype=np.int64))
            np.add.at(Hall, b, 1)
            np.add.at(Hpos, b[posmask], 1)
            np.add.at(Hneg, b[~posmask], 1)
        out = osp.join(exp, f"score_hist_{name}.csv")
        with open(out, "w") as o:
            o.write("bin_left,all,pos,neg\n")
            for i in range(NB):
                o.write(f"{i/NB:.3f},{int(Hall[i])},{int(Hpos[i])},{int(Hneg[i])}\n")
        print(f"[{name}] diseases={len(pos_by_dis)} scores={int(Hall.sum())} positives={int(Hpos.sum())} -> {out}")


if __name__ == "__main__":
    main()
