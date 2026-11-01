# DisGeneFormer
An Attention-based Graph Transformer for Disease Gene Prioritization. 

We describe our approach in the Preprint [“DisGeneFormer: Precise Disease Gene Prioritization by Integrating Local and Global Graph Attention”](https://www.biorxiv.org/content/10.64898/2026.03.11.711106).

Any publication that discloses findings arising from using this source code, the model parameters, or outputs produced by those should also cite this paper.

If you encounter any problems, please open an Issue on this repository.

## Table of contents
* [Setup and Installation](#setup)
    * [Environment](#environment)
    * [Download all datasets](#environment)
* [Usage](#usage)
    * [Train DisGeneFormer](#train-disgeneformer)
    * [Predict disease gene rankings with DisGeneFormer](#predict-disease-gene-rankings-with-disgeneformer)
    * [Evaluate DisGeneFormer's gene rankings against known disease gene associations](#evaluate-disGeneFormer's-gene-rankings-against-known-disease-gene-associations)
* [Reproduce all results](#reproduce-all-results)
    * [Reproduce all results without training](#reproduce-all-results-without-training)
    * [Download all models and results](#download-all-models-and-results)
* [Plot results from the manuscript](#plot-results-from-the-manuscript)
    * [Plot model comparison true positive (TP) curves](#plot-model-comparison-true-positive-tp-curves)
    * [Plot HumanNet comparison TP curves](#plot-humannet-comparison-tp-curves)
    * [Plot Hard Negatives identity scatter plot](#plot-hard-negatives-identity-scatter-plot)
    * [Reproduce graph feature ablation table](#reproduce-graph-feature-ablation-table)

## Setup and Installation

### Environment

DisGeneFormer requires Python 3.11.9
The environment is setup using conda and pip packages.
If there are issues installing the pip packages when setting up the conda environment from `environment/environment.yml`, activate the conda environment and then install them separately using pip while inside the environment.
```bash
conda env create -f environment/environment.yml
conda activate DisGeneFormer_env
pip install -r environment/requirements.txt
```

### Download all datasets
To download all raw datasets needed to reproduce all experiments in the manuscript, use:
```bash
bash scripts/download_all_datasets.sh
```

To run the Model Comparison experiment, the ranked genes lists reported by the other methods must be downloaded using:
```bash
bash scripts/download_model_comparison_results.sh
```

## Usage
In the following steps, we demonstrate how to use DisGeneFormer (DGF), including training it from scratch, using DGF to predicting a ranked list of disease genes given one or a list of diseases, and running evaluation on the ranked genes list predicted against known associated disease genes for the given diseases.

### Train DisGeneFormer
To train DisGeneFormer, you may use the provided config files in the `/results`. The same config file should be used for training, prediction, and evaluation to ensure reproduceability. To train a new DisGeneFormer model based on the configuration, call `train.py` and pass the directory containing the `config.yml` file, as shown below:
```bash
python train.py results/DisGeneFormer 
```
Or to train the version of DisGeneFormer with filtered GeneNet edges 
```bash
python train.py results/DisGeneFormer_filtered
```

Alternatively, you may download the trained model weights for DisGeneFormer:
```bash
bash scripts/download_all_models_and_results.sh --only DGF
```

You may download all results and model weights from the manuscript by referring to the next sections.

### Predict disease gene rankings with DisGeneFormer
To predict ranked genes lists with DisGeneFormer, run one of the following commands. Note that this will use DGF to predict genes for all diseases provided in `data/eval_diseases.tsv`. You may modify this file to add or remove diseases to include in this by adding the UMLS CUIs of diseases.
```bash
python predict_genes_fold.py results/DisGeneFormer --predict-only
```
Or using the filtered version of DisGeneFormer:
```bash
python predict_genes_fold.py results/DisGeneFormer_filtered --predict-only
```


<!-- To also evaluate the ranked genes lists for diseases with known associations in this repo, you may run it without the `--predict-only` flag as shown below.

Note that this method runs inference and then evaluates on all 5 folds of the trained model and then averages over them as was reported for all experiments in the manuscript. 
To run evaluation on existing ranked genes list without the need for any model inference, refer to the step after to use `evaluate.py`. 
```bash
python predict_genes_fold.py results/DisGeneFormer
```
Or the filtered version
```bash
python predict_genes_fold.py results/DisGeneFormer_filtered
``` -->

<!-- ### Predict the top ranked genes using the model
The model produces a list of ranked genes for each given disease defined in `data/eval_diseases.tsv`. Diseases are provided as UMLS CUIs. 

To get a list of ranked disease genes for the diseases defined in `data/eval_diseases.tsv`, run `predict_disease_genes.py <path_to_saved_model>` as done below:
```bash
python predict_genes_fold.py results/DisGeneFormer --predict-only
```
Or the filtered version
```bash
python predict_genes_fold.py results/DisGeneFormer_filtered --predict-only
``` -->

### Evaluate DisGeneFormer's gene rankings against known disease gene associations
To evaluate the ranked genes of a model on the list of evaluation diseases, without requiring training or inference, simply run `python evaluate.py <path_to_ranked_genes>`. This expects a `ranked_genes` folder inside the directory and each ranked genes file should follow the naming convention `<diseaseId>_ranked_genes.tsv` as shown below. To evaluate the average over all model folds as reported in the manuscript, run `evaluate_fold.py` as shown below on the best version of DisGeneFormer: 

```bash
python evaluate_fold.py results/DisGeneFormer
```
Or the filered version
```bash
python evaluate_fold.py results/DisGeneFormer_filtered
```

<!-- ### Filter GeneNet edges 
To filter the edges added to GeneNet based on specific diseases (UMLS CUIs), you may run the following to create a custom filtered version of HumanNet.

For all genes associated with specific diseases

```bash
python scripts/filter_humannet.py data/gene_net/raw/HumanNet-XC-V3.tsv data/test/raw/all_disgenet_associations.tsv data/gene_net/raw/HumanNet-XC-V3_disease_filtered_2.tsv --diseases C0006142 C0009402 C0023893 C0036341 C0376358 C0001973 C0011581 C0860207 C0005586 C3714756
```

For all genes in the DisGeNET associations

```bash
python scripts/filter_humannet.py data/gene_net/raw/HumanNet-XC-V3.tsv data/test/raw/all_disgenet_associations.tsv data/gene_net/raw/HumanNet-XC-V3_disease_filtered.tsv 
``` -->

## Reproduce all results
To reproduce all experiments from scratch in the manuscript, you may run the following script. Note that this includes training each model from scratch and then running inference and evaluation which may take some time. 

<!-- NOTE: there is a total of 15-25 experiments included to reproduce all results in the manuscript which will take ~ -->

To save computational time, skip to the next step to download all models and results to avoid having to run all training and inference from scratch.
```bash
bash scripts/reproduce_all_experiments.sh
```

### Reproduce all results without training

To reproduce all results in the manuscript WITHOUT training any models, run the following script to run inference and evaluation only and use the saved models without training them from scratch. Note that this requires you to have the model weights from either training the model or downloading the trained model weights.
```bash
bash scripts/reproduce_all_results.sh
```

If you already have the ranked genes, you may speed this up even further by just running evaluating the saved ranked genes to skip both the training and the inference steps:
```bash
bash scripts/evaluate_all.sh
```

### Download all models and results
To avoid having to retrain all models from scratch to reproduce all experiments, we additionally provide all trained models and results from inference (ranked genes lists) and evaluation as reported in the manuscript. You may then use the trained model to run inference and evaluation without the need to train them from scratch.

Note that all reported results are the average over 5 training folds and thus all saved models from 5 folds are required to reproduce the results. Downloading all model weights for all folds will require approximately 30 GB of storage space, while all results excluding the model weights are approximately 500 MB. 

<!-- To save storage space, you may instead skip this step and download all results including the ranked genes list in the next step which excludes all the saved models. This approximately reduces the storage requirements from 30 GB to 500 MB. -->
```bash
bash scripts/download_all_models_and_results.sh
```

To only download DisGeneFormer, without all other models used in other experiments, you may use the `--only` flag with the keyword `DGF` by running the following:
```bash
bash scripts/download_all_models_and_results.sh --only DGF
```

Similarly, you may download the results and model weights for the individual experiments using the commands below
```bash
bash scripts/download_all_models_and_results.sh --only DGF_filtered
bash scripts/download_all_models_and_results.sh --only disease_net_feature_removal
bash scripts/download_all_models_and_results.sh --only gene_net_feature_removal
bash scripts/download_all_models_and_results.sh --only humannet_comparison
bash scripts/download_all_models_and_results.sh --only model_comparison
bash scripts/download_all_models_and_results.sh --only negative_comparison
```

<!-- To download only all the results including ranked genes lists without model weights, run the following:
```bash
bash scripts/download_all_results.sh
``` -->

## Plot results from the manuscript

The following can be used to reproduce all plots from the manuscript linked previously.

### Plot model comparison true positive (TP) curves 

First, combine the results for the model comparison to get a table of the top K {5, 20, 50} precision of DisGeneFormer (both the best version and the one trained on the full XC_V3 version of HumanNet) against other DGP methods compared against in the manuscript which are saved in `results/model_comparison` using the following script which can be used to combine any of the results into a single table. The following uses all default values which can be set with flags.

```bash
python plots/scripts/combine_results.py results/model_comparison 
```

Run the following script to plot the number of true positives for each value of K within a range, showing each method as a separate curve and creating a separate plot for each disease to compare different methods on the same disease.

```bash
python plots/scripts/plot_tp_curves.py results/model_comparison --output-dir plots/results/method_comparison_tp_curves --method-names plots/results/method_comparison_tp_curves/method_names.json
```

### Plot HumanNet comparison TP curves
Using the same script, we can plot the results comparing different versions of HumanNet that DisGeneFormer was trained on.

```bash
python plots/scripts/plot_tp_curves.py results/humannet_comparison --output-dir plots/results/humannet_comparison_tp_curves --method-names plots/results/humannet_comparison_tp_curves/method_names.json 
```

### Plot Hard Negatives identity scatter plot
We then plot the identity scatter plot comparing the difference in performance when training on Hard Negatives (HNs) compared to training on randomly generated negative association data (RNs).

```bash
python plots/scripts/plot_identity_scatter.py results/negative_comparison random_negatives hard_negatives --output-dir plots/results/negatives_comparison_identity_scatter --k-value 20 --metric omim_prec
```

### Reproduce graph feature ablation table
To reproduce the results in the manuscript observing the effects of removing individual feature types from the gene and disease graphs, including the table of results, run the following:

```bash
bash scripts/run_feature_removal_experiment.sh
```

