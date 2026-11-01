import collections
import gzip
import logging
import mmap
import os
import sys
import time
import urllib.request
from shutil import copyfile
import torch
import os.path as osp
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
import ast
import pandas as pd
import shutil, pathlib
from collections import defaultdict

class GeneNet(InMemoryDataset):
    class RawFileEnum:
        gene_ontologys = 0
        humannet = 1
        gene_hpo_disease = 2
        gene_expressions = 3
        gene_pathway_associations = 4
        gene_gtex_rna_seq_expressions = 5

    class ProcessedFileEnum:
        gene_id_data_index = 0
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
            humannet_version (str): [XN|XC|XI|FN|CF|PI]
            features_to_use (list): list containing at least one of trace|leaf|hpo|expressions|pathway
        """
        # Set seeds for reproducibility
        self.seed = cfg.get('train', {}).get('seed', 42)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Use deterministic algorithms (increases processing time by 5x)
        # torch.use_deterministic_algorithms(True, warn_only=False)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        self.root = os.path.join(experiment_dir, "gene_net")
        os.makedirs(self.root, exist_ok=True)
        self._raw_root = os.path.join("data/gene_net", "raw")
        os.makedirs(self._raw_root, exist_ok=True)

        self.humannet_version = cfg.gene_net.humannet_version
        self.features_to_use = cfg.gene_net.features_to_use
        self.skip_truncated_svd = cfg.gene_net.skip_truncated_svd
        self.svd_components = cfg.gene_net.svd_components
        self.svd_n_iter = cfg.gene_net.svd_n_iter
        self.train_positives_path = cfg.gene_net.train_positives_path
        self.train_negatives_path = cfg.gene_net.train_negatives_path
        self.all_associations_path = cfg.evaluation.all_disgenet_associations_path
        self.gda_edges = cfg.gene_net.gda_edges
        self.test_path = cfg.evaluation.test_path
        super(GeneNet, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[self.ProcessedFileEnum.data])

    @property
    def raw_dir(self):
        return str(self._raw_root)


    @property
    def raw_file_names(self):
        base_names = [
            'gene_ontologies.tsv',
            'HumanNet-' + self.humannet_version + '.tsv',
            'gene_hpo_disease.tsv', 
            'gene_expressions.tsv',
            'gene_pathway_associations.tsv',
            'gene_gtex_rna_seq_expressions.tsv'
        ]

        return base_names


    @property
    def processed_file_names(self):
        return [
            'gene_id_index_mapping.tsv',
            'edges.pt',
            'nodes.pt',
            'data.pt'
        ]

    def download(self):
        for file in self.raw_file_names:
            dest_file = os.path.join(self.raw_dir, file)
            if not os.path.isfile(dest_file):
                    src = os.path.join(self.raw_dir, '..', '..', file)
                    copyfile(src, dest_file)
    def process(self):
        # Process Edges
        if not os.path.isfile(self.processed_paths[self.ProcessedFileEnum.gene_id_data_index]):
            self.create_node_index_mapping()

        if not os.path.isfile(self.processed_paths[self.ProcessedFileEnum.edges]):
            self.generate_edges()

        # Process node feature matrix.
        if not os.path.isfile(self.processed_paths[self.ProcessedFileEnum.nodes]):
            if 'gtex' in self.features_to_use:
                self.generate_node_feature_matrix_gtex()
            else:
                self.generate_node_feature_matrix()

        # Create and store the data object
        if not os.path.isfile(self.processed_paths[self.ProcessedFileEnum.data]):
            self.generate_data_object()

    @staticmethod
    def download_reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        progress_size_mb = progress_size / (1024 * 1024)
        sys.stdout.write(f'\r {percent}%, {progress_size_mb:.2f} MB, {speed} KB/s, {int(duration)} seconds passed.')
        sys.stdout.flush()

    @staticmethod
    def get_gzip_line_count(in_file, ignore_count=0):
        """ Count the number of lines in a file.

        Args:
            in_file (str):
            ignore_count (int): Remove this from the total count (Ignore headers for example).

        Returns (int): The number of lines in file_path
        """
        count = - ignore_count
        with gzip.open(in_file, 'rt') as f:
            for _ in f:
                count += 1
        return count

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

    def add_generic_features(self, gene_ontology_map, in_file, term_index, gene_id_index=0):
        n_total = self.get_len_file(in_file, ignore_count=0)
        logging.info(f'Creating the feature vectors from `{osp.basename(in_file)}`')
        with open(in_file) as ontology_file:
            for line in tqdm(ontology_file.readlines(), total=n_total):
                parts = [s.strip() for s in line.split('\t')]
                gene_id = int(parts[gene_id_index])
                gene_ontology_map[gene_id].add(parts[term_index])

    def generate_node_feature_matrix(self):
        logging.info('Creating the feature vectors.')
        node_index_mapping = self.load_node_index_mapping()

        # Collect all ontologys from the raw file and create a gene ontology map.
        gene_ontology_map = collections.defaultdict(set)
        if 'leaf' in self.features_to_use:
            self.add_generic_features(
                gene_ontology_map,
                in_file=self.raw_paths[self.RawFileEnum.gene_ontologys],
                term_index=1,
                gene_id_index=0
            )
        if 'trace' in self.features_to_use:
            self.add_generic_features(
                gene_ontology_map,
                in_file=self.raw_paths[self.RawFileEnum.gene_ontologys],
                term_index=3,
                gene_id_index=0
            )
        if 'hpo' in self.features_to_use:
            self.add_generic_features(
                gene_ontology_map,
                in_file=self.raw_paths[self.RawFileEnum.gene_hpo_disease],
                term_index=1,
                gene_id_index=0
            )
        if 'expressions' in self.features_to_use:
            self.add_generic_features(
                gene_ontology_map,
                in_file=self.raw_paths[self.RawFileEnum.gene_expressions],
                term_index=1,
                gene_id_index=0
            )
        if 'pathways' in self.features_to_use:
            self.add_generic_features(
                gene_ontology_map,
                in_file=self.raw_paths[self.RawFileEnum.gene_pathway_associations],
                term_index=1,
                gene_id_index=0
            )

        # Collect all appearing terms and enumerate them in the ontology_index_map
        all_ontologys = set()
        for asc_ontologys in gene_ontology_map.values():
            for ontology in asc_ontologys:
                all_ontologys.add(ontology)
        all_ontologys = sorted(all_ontologys)
        ontology_index_map = {ont: ind for (ind, ont) in enumerate(all_ontologys)}

        # Create the feature vectors by one hot encode the associated ontology terms.
        row_ind, col_ind, scores = [], [], []
        for gene_id, ontologys in sorted(gene_ontology_map.items()):
            if gene_id in node_index_mapping:
                for ontology in sorted(ontologys):
                    row_ind.append(int(node_index_mapping[gene_id]))
                    col_ind.append(int(ontology_index_map[ontology]))
                    score = 1  # One Hot encoding
                    scores.append(score)

        x = coo_matrix((scores, (row_ind, col_ind)), shape=(len(node_index_mapping), len(all_ontologys)))

        if not self.skip_truncated_svd:
            logging.info('Computing TruncatedSVD')
            svd = TruncatedSVD(n_components=self.svd_components, n_iter=self.svd_n_iter, random_state=self.seed)
            svd.fit(x)
            x = svd.transform(x)
            x = torch.tensor(x).float()
        else:
            logging.info('Skipped TruncatedSVD')
            x = torch.tensor(x.toarray()).float()

        torch.save(x, self.processed_paths[self.ProcessedFileEnum.nodes])

    def generate_node_feature_matrix_gtex(self):
        # Extract expression score vectors from tissue expression levels according to the GTex RNA-seq data.
        logging.info('Creating gtex feature vectors.')
        node_index_mapping = self.load_node_index_mapping()

        # Collect all ontologys from the raw file and create a gene ontology map
        expression_scores = collections.defaultdict(set)
        n_total = self.get_len_file(in_file, ignore_count=0)
        logging.info(f'Creating the feature vectors from `{osp.basename(in_file)}`')
        with open(self.raw_paths[self.RawFileEnum.gene_gtex_rna_seq_expressions]) as input_file:
            for line in tqdm(input_file.readlines(), total=n_total):
                parts = [s.strip() for s in line.split('\t')]
                gene_id = int(parts[0])

                expression_scores[gene_id].add(parts[term_index])

        # Collect all appearing terms and enumerate them in the ontology_index_map
        all_ontologys = set()
        all_ontologys = sorted(all_ontologys)
        for asc_ontologys in gene_ontology_map.values():
            for ontology in asc_ontologys:
                all_ontologys.add(ontology)
        ontology_index_map = {ont: ind for (ind, ont) in enumerate(all_ontologys)}

        # Create the feature vectors by one hot encode the associated ontology terms.
        row_ind, col_ind, scores = [], [], []
        for gene_id, ontologys in sorted(gene_ontology_map.items()):
            if gene_id in node_index_mapping:
                for ontology in sorted(ontologys):
                    row_ind.append(int(node_index_mapping[gene_id]))
                    col_ind.append(int(ontology_index_map[ontology]))
                    score = 1  # One Hot encoding
                    scores.append(score)

        x = coo_matrix((scores, (row_ind, col_ind)), shape=(len(node_index_mapping), len(all_ontologys)))

        if not self.skip_truncated_svd:
            logging.info('Computing TruncatedSVD')
            svd = TruncatedSVD(n_components=self.svd_components, n_iter=self.svd_n_iter, random_state=self.seed)
            svd.fit(x)
            x = svd.transform(x)
            x = torch.tensor(x).float()
        else:
            logging.info('Skipped TruncatedSVD')
            x = torch.tensor(x.toarray()).float()

        torch.save(x, self.processed_paths[self.ProcessedFileEnum.nodes])

    def load_edges(self):
        return torch.load(self.processed_paths[self.ProcessedFileEnum.edges])

    def generate_edges(self):
        node_index_mapping = self.load_node_index_mapping()
        sources, targets, scores = [], [], []

        logging.info('Generating the gene edges from HumanNet.')
        with open(osp.join(self.raw_dir, self.raw_file_names[self.RawFileEnum.humannet])) as gene_links_file:
            for gene_link in gene_links_file.readlines():
                if gene_link.startswith('#'):
                    continue
                source, target, score = [s.strip() for s in gene_link.split()]
                scores.append(float(score))
                sources.append(node_index_mapping[int(source)])
                targets.append(node_index_mapping[int(target)])

        if self.gda_edges:
            print("Using GDA edges for GeneNet")
            disease_genes = pd.read_csv(self.train_positives_path, sep='\t', names=['EntrezGene ID', 'disease_id', 'Score'])
            gene_to_diseases = defaultdict(set)
            for _, row in disease_genes.iterrows():
                gene_id = int(row['EntrezGene ID'])
                if gene_id in node_index_mapping:  # Only add if gene is in our mapping
                    gene_to_diseases[gene_id].add(row['disease_id'])
                    
            # Connect genes sharing diseases
            for gene1, diseases1 in sorted(gene_to_diseases.items()):
                for gene2, diseases2 in sorted(gene_to_diseases.items()):
                    if gene1 < gene2:  
                        # Weight edges between genes by 1 
                        shared = 1
                        if shared > 0:
                            sources.append(node_index_mapping[gene1])
                            targets.append(node_index_mapping[gene2])
                            scores.append(shared)  # Weight by number of shared diseases
        
        
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_attr = torch.tensor(scores).reshape((len(sources), 1))


        print(f"Number of Gene edges: {len(edge_attr)}")

        torch.save((edge_index, edge_attr), self.processed_paths[self.ProcessedFileEnum.edges])

    def _collect_gene_ids_from_features(self) -> list[int]:
        """Return every Entrez ID that occurs in any raw feature file."""
        genes: set[int] = set()
        use  = self.features_to_use
        R    = self.raw_paths 

        def grab(path, gid_idx):
            with open(path) as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if parts and parts[0].isdigit():
                        genes.add(int(parts[gid_idx]))

        if 'leaf' in use or 'trace' in use:
            grab(R[self.RawFileEnum.gene_ontologys], 0)       
        if 'hpo' in use:
            grab(R[self.RawFileEnum.gene_hpo_disease], 0)
        if 'expressions' in use:
            grab(R[self.RawFileEnum.gene_expressions], 0)
        if 'pathways' in use:
            grab(R[self.RawFileEnum.gene_pathway_associations], 0)
            print(f"pathways used: {len(genes)} genes found")
        if 'gtex' in use:
            grab(R[self.RawFileEnum.gene_gtex_rna_seq_expressions], 0)
        return sorted(genes)

    def create_node_index_mapping(self):
        """ Creates a mapping between gene_id and index to be used in the data tensor.
        Stores the resut to 'gene_id_index_mapping.tsv'

        """
        node_index_mapping: dict[str, int] = collections.OrderedDict()

        def add_gene(gid: str | int):
            gid = str(gid)
            if gid not in node_index_mapping:
                node_index_mapping[gid] = len(node_index_mapping)

        # 1) positives ------------------------------------------------------
        # DisGeNET associations have a header row; the gene id is the first column.
        pos = pd.read_csv(self.all_associations_path, sep="\t").iloc[:, 0].dropna().tolist()
        for gid in sorted(pos, key=int):                    
            add_gene(gid)

        # 2) negatives ------------------------------------------------------
        if self.train_negatives_path != "random":
            neg = pd.read_csv(self.train_negatives_path, sep="\t", header=None)[0].dropna().tolist()
            for gid in sorted(neg, key=int):
                add_gene(gid)

        # 3) genes from feature files --------------------------------------
        for gid in sorted(self._collect_gene_ids_from_features()):
            add_gene(gid)

        # 4) genes from HumanNet -------------------------------------------
        with open(self.raw_paths[self.RawFileEnum.humannet]) as fh:
            nodes = set()
            for ln in fh:
                if ln.startswith("#"):
                    continue
                s, t, *_ = ln.split()
                nodes.update((s, t))
        for gid in sorted(nodes, key=int):
            add_gene(gid)

        # write to disk -----------------------------------------------------
        with open(self.processed_paths[self.ProcessedFileEnum.gene_id_data_index], "w") as out:
            out.write("{gene_id}\t{index}\n")  # header
            for gid, idx in sorted(node_index_mapping.items()):
                out.write(f"{gid}\t{idx}\n")

    def load_node_index_mapping(self):
        node_index_mapping = collections.OrderedDict()
        with open(self.processed_paths[self.ProcessedFileEnum.gene_id_data_index], mode='r') as file:
            next(file)
            for line in file.readlines():
                gene_id, index = [int(s.strip()) for s in line.split('\t')]
                node_index_mapping[gene_id] = index
        print(f"node index mapping: {len(node_index_mapping)}")
        return node_index_mapping


if __name__ == '__main__':
    HERE = osp.abspath(osp.dirname(__file__))
    DATASET_ROOT = osp.join(HERE, 'data_sources', 'dataset_gene_network_test')

    gene_net = GeneNet(root=DATASET_ROOT)
