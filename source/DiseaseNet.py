import collections
import gzip
import itertools
import logging
import mmap
import os
import os.path as osp
import sys
import torch

from shutil import copyfile

from sklearn import neighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

from collections import defaultdict

import pandas as pd
import shutil, pathlib


class DiseaseNet(InMemoryDataset):
    class RawFileEnum:
        # Indices MUST match the order of raw_file_names below.
        disease_hpo = 0
        disease_publication_titles = 1
        all_diseases = 2
        disease_pathway = 3
        disease_umls = 4          # UMLS semantic-type + relation features (Option A)

    class ProcessedFileEnum:
        disease_id_feature_index_mapping = 0
        edges = 1
        nodes = 2
        data = 3

    def __init__(
            self,
            cfg,
            experiment_dir,
            transform=None,
            pre_transform=None,
    ):
        """

        Args:
            root:
            transform:
            pre_transform:
            edge_source (str): Out of {'databases', 'feature_similarity}. 'databases' will use shared database features
                such as shared pathway etc. 'feature_similarity' will use a kNN approach to create disease links.
            hpo_count_freq_cutoff (int): Consider only disease ontology terms associated to less than
                hpo_count_freq_cutoff diseases for building edges.
            feature_source (list): List of which sources to use to create the disease feature vectors.
                Out of {'disease_publications', 'phenotypes'}
            n_neighbours (int): It the edge_source is set to feature_similarity: Number of most similar nodes to
            consider.
        """
        self.root = os.path.join(experiment_dir, "disease_net")
        os.makedirs(self.root, exist_ok=True)
        # Raw disease files live here. Config-driven (disease_net.raw_root) so a run can
        # point at CUI-keyed features, e.g. data/disease_net/raw_cui.
        self._raw_root = cfg.disease_net.get('raw_root', os.path.join("data/disease_net", "raw"))
        os.makedirs(self._raw_root, exist_ok=True)
        # Which UMLS feature file to read for the 'umls' source (lets enriched and
        # unenriched builds coexist and be selected per-config).
        self._umls_file = cfg.disease_net.get('umls_file', 'disease_umls.tsv')
        # Option B: edge list file for edge_source == 'umls_relations'.
        self._umls_edges_file = cfg.disease_net.get('umls_edges_file', 'disease_umls_edges.tsv')

        self.skip_truncated_svd = cfg.disease_net.skip_truncated_svd
        self.svd_components = cfg.disease_net.svd_components
        self.svd_n_iter = cfg.disease_net.svd_n_iter
        self.edge_source = cfg.disease_net.edge_source
        self.feature_source = cfg.disease_net.features_to_use
        self.hpo_count_freq_cutoff = cfg.disease_net.hpo_count_freq_cutoff
        self.n_neighbours = cfg.disease_net.n_neighbors
        self.train_positives_path = cfg.disease_net.train_positives_path
        self.train_negatives_path = cfg.disease_net.train_negatives_path
        self.gda_edges = cfg.disease_net.gda_edges
        self.test_path = cfg.evaluation.test_path
        # Which disease-id prefixes to admit from all_diseases.tsv. Disease ids are
        # UMLS CUIs (prefix 'C'); [] / ['*'] admits every id.
        self.disease_id_prefixes = tuple(cfg.disease_net.get('id_prefixes', ['C']))
        super(DiseaseNet, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[self.ProcessedFileEnum.data])
        self.seed = cfg.get('train', {}).get('seed', 42)

        # Set seeds for reproducibility
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
    @property
    def raw_dir(self):
        return str(self._raw_root)

    @property
    def raw_file_names(self):

        base_names = [
            'disease_hpo.tsv',
            'disease_publication_titles_and_abstracts.tsv',
            'all_diseases.tsv',
            'disease_pathway.tsv',
            'disease_umls.tsv'      # order MUST match RawFileEnum above
        ]

        return base_names

    @property
    def processed_file_names(self):
        return [
            'disease_id_feature_index_mapping.tsv',
            'edges.pt',
            'nodes.pt',
            'data.pt'
        ]

    def download(self):
        for file in self.raw_file_names:
            dest = os.path.join(self.raw_dir, file)
            if not os.path.isfile(dest):
                src = os.path.join(self.raw_dir, '..', '..', file)
                # An optional raw file may be absent (e.g. disease_hpo.tsv when using
                # UMLS features instead, or disease_umls.tsv on a classic HPO run).
                # Skip it here; process() errors only if it is actually needed by the
                # configured features_to_use.
                if os.path.isfile(src):
                    copyfile(src, dest)

    def process(self):
        logging.info('Create disease_id feature_index mapping.')
        self.create_disease_index_feature_mapping()

        logging.info('Create feature matrix.')
        self.generate_disease_feature_matrix()

        logging.info('Create edges.')
        if self.edge_source == 'databases':
            self.generate_edges()
        if self.edge_source == 'feature_similarity':
            self.generate_edges_similarity_based()
        if self.edge_source == 'umls_relations':
            self.generate_edges_umls_relations()

        # Create and store the data object
        self.generate_data_object()

    @staticmethod
    def get_len_file(in_file, ignore_count=0):
        """ Count the number of lines in a file.

        Args:
            in_file (str):
            ignore_count (int): Remove this from the total count (Ignore headers for example).

        Returns (int): The number of lines in file_path
        """
        fp = open(in_file, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines - ignore_count

    def generate_data_object(self):
        x = self.load_node_feature_martix()
        edge_index, edge_attr = self.load_edges()
        data_list = [
            Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        logging.info('Storing the data.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.ProcessedFileEnum.data])
        logging.info('Done.')

    def load_node_feature_martix(self):
        return torch.load(self.processed_paths[self.ProcessedFileEnum.nodes])

    def generate_disease_feature_matrix(self):
        logging.info('Creating disease feature vectors.')

        disease_index_mapping = self.load_disease_index_feature_mapping()
        x = None
        if 'phenotypes' in self.feature_source:
            logging.info('Create phenotype feature vectors.')
            mlb = MultiLabelBinarizer()
            hpo_ids = set()
            disease_hpo_map = collections.defaultdict(set)
            with open(osp.join(self.raw_dir, self.raw_file_names[self.RawFileEnum.disease_hpo])) as disease_hpo_file:
                for disease_link in disease_hpo_file.readlines():
                    dis_id, hpo_id, hpo_name = [s.strip() for s in disease_link.split('\t')]
                    hpo_ids.add(hpo_id)
                    disease_hpo_map[dis_id].add(hpo_id)

            mlb.fit([sorted(hpo_ids)])
            disease_id_sorted_by_index = sorted(disease_index_mapping.keys(), key=lambda x: disease_index_mapping[x])
            disease_features = [sorted(disease_hpo_map[d_id]) for d_id in disease_id_sorted_by_index]
            # Create the feature matrix
            x = torch.tensor(mlb.transform(disease_features), dtype=torch.float)

        if 'pathways' in self.feature_source:
            logging.info('Create pathway feature vectors.')
            mlb = MultiLabelBinarizer()
            pathway_ids = set()
            disease_pathway_map = collections.defaultdict(set)
            with open(osp.join(self.raw_dir, self.raw_file_names[self.RawFileEnum.disease_pathway])) as file:
                for pathway_link in file.readlines():
                    dis_id, pathway_id = [s.strip() for s in pathway_link.split('\t')]
                    pathway_ids.add(pathway_id)
                    disease_pathway_map[dis_id].add(pathway_id)

            mlb.fit([sorted(pathway_ids)])
            disease_id_sorted_by_index = sorted(disease_index_mapping.keys(), key=lambda x: disease_index_mapping[x])
            disease_features = [sorted(disease_pathway_map[d_id]) for d_id in disease_id_sorted_by_index]
            # Create the feature matrix
            tmp = torch.tensor(mlb.transform(disease_features), dtype=torch.float)
            # Concat with previous feature matrix.
            if x is not None:
                x = torch.cat((x, tmp), dim=1)
            else:
                x = tmp

        if 'umls' in self.feature_source:
            logging.info('Create UMLS (semantic-type + relation) feature vectors.')
            mlb = MultiLabelBinarizer()
            umls_tokens = set()
            disease_umls_map = collections.defaultdict(set)
            with open(osp.join(self.raw_dir, self._umls_file)) as umls_file:
                for line in umls_file:
                    row = [s.strip() for s in line.split('\t')]
                    if len(row) < 2:
                        continue
                    dis_id, token = row[0], row[1]
                    umls_tokens.add(token)
                    disease_umls_map[dis_id].add(token)

            mlb.fit([sorted(umls_tokens)])
            disease_id_sorted_by_index = sorted(disease_index_mapping.keys(), key=lambda x: disease_index_mapping[x])
            disease_features = [sorted(disease_umls_map[d_id]) for d_id in disease_id_sorted_by_index]
            tmp = torch.tensor(mlb.transform(disease_features), dtype=torch.float)
            if x is not None:
                x = torch.cat((x, tmp), dim=1)
            else:
                x = tmp

        if 'disease_publications' in self.feature_source:
            logging.info('Create publication feature vectors.')
            disease_id_publication_titles = collections.defaultdict(str)
            corpus = []
            with open(
                self.raw_paths[self.RawFileEnum.disease_publication_titles],
                mode='r',
                encoding='utf-8'
            ) as disease_publications:
                for line in disease_publications:
                    disease_id, publication_title, publication_abstract = [s.strip() for s in line.split('\t')]
                    if len(publication_abstract) > 0:
                        corpus.append(publication_abstract)
                        disease_id_publication_titles[disease_id] += f' {publication_abstract}'
                    if len(publication_title) > 0:
                        corpus.append(publication_title)
                        disease_id_publication_titles[disease_id] += f' {publication_title}'

            # Build the vectorizer
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_df=0.01,  # occurs in max 10% of the diseases.
                min_df=0.001  # occurs in at least 0.1% of the diseases.
            )
            vectorizer.fit(corpus)

            # Create the feature matrix
            tmp = torch.tensor(vectorizer.transform(
                [disease_id_publication_titles[oid] for oid in
                 sorted(disease_index_mapping.keys(), key=lambda x: disease_index_mapping[x])]
            ).toarray(), dtype=torch.float)
            # Concat with previous feature matrix.
            if x is not None:
                x = torch.cat((x, tmp), dim=1)
            else:
                x = tmp

        if not self.skip_truncated_svd:
            logging.info('Doing dimensionality reduction using TruncatedSVD')
            svd = TruncatedSVD(n_components=self.svd_components, n_iter=self.svd_n_iter, random_state=self.seed)
            svd.fit(x)
            x = svd.transform(x)
            x = torch.tensor(x).float()

        torch.save(x, self.processed_paths[self.ProcessedFileEnum.nodes])

    def load_edges(self):
        return torch.load(self.processed_paths[self.ProcessedFileEnum.edges])

    def generate_edges(self):
        disease_index_mapping = self.load_disease_index_feature_mapping()
        to_be_linked_diseases = collections.defaultdict(set)

        logging.info('Generate disease edges using gene-disease association data')
        gene_diseases = pd.read_csv(self.train_positives_path, sep='\t', names=['EntrezGene ID', 'disease_id', 'Score'])
        disease_to_genes = defaultdict(set)
        for _, row in gene_diseases.iterrows():
            disease_to_genes[row['disease_id']].add(row['EntrezGene ID'])
        
        if self.gda_edges:
            # Connect diseases sharing genes
            disease_ids = sorted(disease_to_genes.keys())
            for i, dis1 in enumerate(disease_ids):
                genes1 = disease_to_genes[dis1]
                for dis2 in disease_ids[i+1:]:
                    genes2 = disease_to_genes[dis2]
                    shared = len(genes1 & genes2)
                    if shared:
                        sources.append(disease_index_mapping[dis1])
                        targets.append(disease_index_mapping[dis2])
                        scores.append(shared)  # Weight by number of shared genes
            for disease1, genes1 in sorted(disease_to_genes):
                for disease2, genes2 in sorted(disease_to_genes.items()):
                    if disease1 < disease2:
                        shared = len(genes1 & genes2)
                        if shared > 0:
                            sources.append(disease_index_mapping[disease1])
                            targets.append(disease_index_mapping[disease2]) 
                            scores.append(shared)  # Weight by number of shared genes

        logging.info('Generating the disease edges.')
        logging.info('Using shared phenotypes.')
        with open(osp.join(self.raw_dir, self.raw_file_names[self.RawFileEnum.disease_hpo])) as disease_hpo_file:
            for disease_link in disease_hpo_file.readlines():
                dis_id, hpo_id, hpo_name = [s.strip() for s in disease_link.split('\t')]
                to_be_linked_diseases[hpo_id].add(disease_index_mapping[dis_id])

        logging.info('Using shared pathways.')
        with open(osp.join(self.raw_dir, self.raw_file_names[self.RawFileEnum.disease_pathway])) as file:
            for pathway_link in file.readlines():
                dis_id, pathway_id = [s.strip() for s in pathway_link.split('\t')]
                to_be_linked_diseases[pathway_id].add(disease_index_mapping[dis_id])

        edges = set()
        len_counts = collections.defaultdict(int)
        for key in sorted(to_be_linked_diseases.keys()):
            diseases = to_be_linked_diseases[key]
            len_counts[len(diseases)] += 1
            if len(diseases) > self.hpo_count_freq_cutoff:
                continue
            for source, target in itertools.combinations(diseases, 2):
                edges.add((source, target))

        sources, targets, scores = [], [], []
        for source, target in edges:
            scores.append(1)
            sources.append(source)
            targets.append(target)

        edge_tuples = sorted(zip(sources, targets, scores))
        if edge_tuples:
            sources, targets, scores = zip(*edge_tuples)
        else:
            sources, targets, scores = [], [], []

        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_attr = torch.tensor(scores).reshape((len(sources), 1))

        torch.save((edge_index, edge_attr), self.processed_paths[self.ProcessedFileEnum.edges])

    def generate_edges_similarity_based(self):
        logging.info("Generate similarity based edges.")
        X = self.load_node_feature_martix()
        X = normalize(X) 
        adj = neighbors.kneighbors_graph(
            X,
            n_neighbors=self.n_neighbours,
            include_self=True
        ).toarray()
        edge_index, edge_attr = dense_to_sparse(torch.tensor(adj))
        order = edge_index[0].argsort(stable=True)
        edge_index = edge_index[:, order]
        edge_attr = edge_attr[order]

        torch.save((edge_index, edge_attr), self.processed_paths[self.ProcessedFileEnum.edges])

    def generate_edges_umls_relations(self):
        """Option B: disease edges ARE the UMLS hierarchical relations between disease CUIs
        (read from the precomputed edge file), instead of feature-similarity kNN. Nodes with
        no relation edge fall back to feature-similarity kNN so the graph stays connected."""
        import numpy as np
        logging.info('Generate UMLS-relation disease edges (Option B).')
        disease_index_mapping = self.load_disease_index_feature_mapping()
        sources, targets, scores = [], [], []
        seen = set()
        edge_path = osp.join(self.raw_dir, self._umls_edges_file)
        with open(edge_path) as f:
            for line in f:
                parts = [s.strip() for s in line.split('\t')]
                if len(parts) < 2:
                    continue
                c1, c2 = parts[0], parts[1]
                if c1 in disease_index_mapping and c2 in disease_index_mapping:
                    i, j = disease_index_mapping[c1], disease_index_mapping[c2]
                    for a, b in ((i, j), (j, i)):   # undirected
                        if (a, b) not in seen:
                            seen.add((a, b)); sources.append(a); targets.append(b); scores.append(1)

        # Fallback: feature-kNN edges for nodes with no relation edge (avoid isolated nodes).
        X = normalize(np.asarray(self.load_node_feature_martix()))
        n = X.shape[0]
        linked = set(sources)
        isolated = [i for i in range(n) if i not in linked]
        if isolated:
            logging.info(f'{len(isolated)} isolated nodes -> add feature-kNN fallback edges.')
            nn = neighbors.NearestNeighbors(n_neighbors=self.n_neighbours).fit(X)
            _, idxs = nn.kneighbors(X[isolated])
            for row, i in zip(idxs, isolated):
                for j in row:
                    sources.append(i); targets.append(int(j)); scores.append(1)

        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_attr = torch.tensor(scores).reshape((len(sources), 1))
        order = edge_index[0].argsort(stable=True)
        edge_index = edge_index[:, order]
        edge_attr = edge_attr[order]
        logging.info(f'UMLS-relation graph: {edge_index.shape[1]} directed edges over {n} nodes.')
        torch.save((edge_index, edge_attr), self.processed_paths[self.ProcessedFileEnum.edges])

    def create_disease_index_feature_mapping(self):
        """ Creates a mapping between disease and index to be used in the feature matrix.
        Stores the result to disease_id_feature_index_mapping

        """
        disease_index_mapping = collections.OrderedDict()

        with open(self.raw_paths[self.RawFileEnum.all_diseases], mode='r') as in_file:
            for line in in_file:
                parts = [s.strip() for s in line.split('\t')]
                identifier = parts[0]
                accept = (not self.disease_id_prefixes or '*' in self.disease_id_prefixes
                          or identifier.startswith(self.disease_id_prefixes))
                if accept and identifier not in disease_index_mapping:
                    disease_index_mapping[identifier] = len(disease_index_mapping)

        disease_paths = [
            self.train_positives_path,
            self.train_negatives_path
        ]

        for disease_path in disease_paths:
            if not disease_path == 'random':
                new_diseases = pd.read_csv(disease_path, sep='\t', header=None)
                new_diseases = new_diseases[1].drop_duplicates().tolist()

                for disease in new_diseases:
                    if disease not in disease_index_mapping:
                        disease_index_mapping[disease] = len(disease_index_mapping)


        with open(self.processed_paths[self.ProcessedFileEnum.disease_id_feature_index_mapping], mode='w') as out_file:
            out_file.write('{disease_id}\t{index}\n')
            for gene_id, index in disease_index_mapping.items():
                out_file.write(f'{gene_id}\t{index}\n')

    def load_disease_index_feature_mapping(self):
        disease_index_mapping = collections.OrderedDict()
        with open(self.processed_paths[self.ProcessedFileEnum.disease_id_feature_index_mapping], mode='r') as file:
            next(file)
            for line in file.readlines():
                disease_id, index = [s.strip() for s in line.split('\t')]
                index = int(index)
                disease_index_mapping[disease_id] = index

        return disease_index_mapping


if __name__ == '__main__':
    HERE = osp.abspath(osp.dirname(__file__))
    DATASET_ROOT = osp.join(HERE, 'data_sources', 'dataset_diseases')

    disease_net = DiseaseNet(root=DATASET_ROOT)
