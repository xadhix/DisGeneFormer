# DisGeneFormer — Modifications

What this fork changes versus the canonical DisGeneFormer (commit `72bb8b9`, on a
`1c909ee` git base). **Every change is config-gated and the defaults reproduce the
original classification behaviour exactly** — a stock `default_config.yml` trains
and evaluates the same model as before.

> Upstream `github.com/ivelet/DisGeneFormer` was force-pushed to a broken,
> history-less squash (missing `source/HeteroDisEncoder.py`). **Do not `git pull`
> the deployed copy.** The good tree is the `72bb8b9` content this repo is based on.

---

## 1. Regression task — `model.task: classification | regression`

Predict the DisGeNET association **score** instead of a binary class.

| File | Change |
|---|---|
| `source/DisGeneFormer.py:80,87,138` | `self.task`; `self.out_dim = 1 if regression else 2`; final head is `nn.Linear(fc_hidden_dim//2, self.out_dim)`. |
| `train.py:321–348` | Positive target = the score in the positives file's 3rd column (`y`, float), falling back to `1.0` if absent. **Negatives are always `0`** in both tasks. `y_out` is `float32` for regression, `long` for classification. |
| `train.py:364–379` | `regression_loss` = `sigmoid(out[:,0])` then `F.mse_loss` (or `F.smooth_l1_loss` when `loss_fn ∈ {smooth_l1, huber}`). |
| `train.py:373–378` | `criterion` select: regression → `regression_loss`; else `cross_entropy` or `focal_loss`. |
| `predict_genes_fold.py:317–321` | Regression ranks genes by `sigmoid(logits[:,0])`; classification by `logits[:,1]` (unchanged). |

Training-file column order is **gene, disease, score** (`EntrezGene ID`, `OMIM ID`
or CUI, `Score`). Without a scored file it still runs as binary-float regression.

## 2. Transformer ablation — `model.ablate.global_transformer: False | True`

Compare the model **WITH vs WITHOUT** the final 2-layer global transformer. This is
the only ablation switch; it is on/off, **not** a layer-count sweep.

| File | Change |
|---|---|
| `source/DisGeneFormer.py:83–84` | `self.ablate = dict(ablate or {})`; `self.use_global_transformer = not self.ablate.get('global_transformer', False)`. |
| `source/DisGeneFormer.py:192–193, 254–255` | Both call sites gated: `if self.use_global_transformer: combined = self.global_transformer(combined)`. The layer is still **built** (line 123) so checkpoints stay shape-compatible; it just isn't applied. |

## 3. Config keys that were dead, now wired

These existed in the config but were ignored; they now take effect.

| Key | Effect | Where |
|---|---|---|
| `model.transformer_num_heads / _num_layers / _dropout` | fed into `GlobalGraphTransformer` (were hardcoded 4 / 2 / 0.1) | `DisGeneFormer.py:123–127`, `train.py` reads at 113–115 (passed to the model) |
| `train.loss_fn` | `focal_loss`\|`cross_entropy` (cls); `mse`\|`smooth_l1` (reg) | `train.py:109, 373–378` |
| `train.optimizer` | `Adam`\|`AdamW` | `train.py:110, 440–441` |
| `train.grad_clip_max_norm` | active in the training loop (autocast + non-autocast paths); `0` disables | `train.py:111, 472–482` |

Note: `grad_clip_max_norm: 5` is now **active**. Set `0` to match the pre-change
baseline exactly (previously it was a no-op call in the model constructor, removed).

## 4. CUI (non-OMIM) disease ids — `disease_net.id_prefixes`

DisGeneFormer is id-agnostic (the disease id is an opaque key, auto-registered from
the data file). The only hardcoded OMIM assumption was a `startswith('OMIM')` filter.

| File | Change |
|---|---|
| `source/DiseaseNet.py:83` | `self.disease_id_prefixes = tuple(cfg.disease_net.get('id_prefixes', ['OMIM']))`. |
| `source/DiseaseNet.py:371–373` | `accept = (not prefixes) or ('*' in prefixes) or identifier.startswith(prefixes)` — replaces the hardcoded OMIM check. |

Default `['OMIM']`; use `['C']` (or `['UMLS']`) for CUIs, `['*']`/`[]` for all. A
`(geneId, CUI, score)` training file then runs with `task: regression` and **no
further model code**. CUI disease *features* are produced by re-keying the OMIM
feature files — see `dgf_rekey_disease_features.py` below.

## 5. Decided out: no `-1` negative target

Regression negatives are always fit as `0` ("no association"). The earlier
config-gated `negative_target = -1` masking branch **and** the `data.negative_target`
key were removed. (Behaviourally inert for past runs: the default was already `0`.)

## Still NOT wired — setting these does nothing

`train.attn_dropout`, `model.transformer_hidden_dim`, `model.num_attn_heads`,
`model.num_layers` (the GAT side is hardcoded to 3 layers / 4 heads in
`AttentionAggregation`, `DisGeneFormer.py:9`).

## Config quick reference (`default_config.yml`)

```yaml
model:
  task: 'classification'            # or 'regression'
  transformer_num_heads: 4          # now wired
  transformer_num_layers: 2         # now wired
  transformer_dropout: 0.1          # now wired
  ablate:
    global_transformer: False       # True -> WITHOUT the final transformer
disease_net:
  id_prefixes: ['OMIM']             # ['C'] for CUIs, ['*'] for all
train:
  loss_fn: 'focal_loss'             # mse|smooth_l1 for regression
  optimizer: 'Adam'                 # or AdamW
  grad_clip_max_norm: 5             # now active; 0 = off
# (data.negative_target removed — negatives are always 0)
```

---

## Driving scripts (live in the sibling `syndata/scripts/`)

These orchestrate the above on bwForCluster Helix. Full commands: `syndata/DGF_RUNBOOK.md`.

| Script | Role |
|---|---|
| `dgf_sweep.sh` | **One command** to run the whole matrix: generates the configs and submits one independent job per cell. `bash dgf_sweep.sh` → task × transformer{WITH,WITHOUT} = 4 jobs; `--lrs ...` adds an lr axis; `--check` dry-runs. |
| `dgf_make_configs.py` | Generates the config matrix (`--tasks`, `--transformer on/off`, `--lrs`) from `default_config.yml`. Renders lrs YAML-float-safe (`1e-4` alone loads as a string in PyYAML). |
| `helix_jobs.py` | `sweep` (one isolated job per config across A40/A100/H200), plus `submit`/`status`/`keep`/`cancel`. Each job gets its own `results/helix/<group>/<label>/`, `logs/.../<label>/`, and `config.yml`. |
| `dgf_eval_diseases.slurm` | Score an arbitrary disease list: `train → predict → evaluate` into a per-SLURM-job experiment dir (`--train` to build a fresh model). |
| `dgf_rekey_disease_features.py` | Re-keys the OMIM-keyed `disease_hpo.tsv` / `disease_pathway.tsv` to CUI via syndata's MRCONSO OMIM↔CUI crosswalk, so CUI-space runs are first-class. `--restrict-universe` (default on) keeps only disease/phenotype CUIs. |

### Non-interference (why parallel jobs are safe)

`GeneNet` and `DiseaseNet` root their processed graphs **under the experiment dir**
(`GeneNet.py:64`, `DiseaseNet.py:63`), and train/predict/evaluate all read and write
only there. The sweep payload (`dgf_helix.slurm`) touches no shared writable path, so
N sweep jobs run fully in parallel. `dgf_eval_diseases.slurm` additionally appends the
SLURM job id to its default experiment dir; its one shared mutable file,
`data/eval_diseases.tsv`, is safe for identical lists but would race if two concurrent
jobs rank **different** lists.
