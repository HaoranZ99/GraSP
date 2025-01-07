import glob
import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
    extract_zip,
)
from torch_geometric.utils import to_undirected
from utils import random_walk_positional_encoding
import random

class MyGEDDataset(InMemoryDataset):
    r"""The GED datasets from the `"Graph Edit Distance Computation via Graph
    Neural Networks" <https://arxiv.org/abs/1808.05689>`_ paper.
    GEDs can be accessed via the global attributes :obj:`ged` and
    :obj:`norm_ged` for all train/train graph pairs and all train/test graph
    pairs:

    .. code-block:: python

        dataset = GEDDataset(root, name="LINUX")
        data1, data2 = dataset[0], dataset[1]
        ged = dataset.ged[data1.i, data2.i]  # GED between `data1` and `data2`.

    Note that GEDs are not available if both graphs are from the test set.
    For evaluation, it is recommended to pair up each graph from the test set
    with each graph in the training set.

    .. note::

        :obj:`ALKANE` is missing GEDs for train/test graph pairs since they are
        not provided in the `official datasets
        <https://github.com/yunshengb/SimGNN>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (one of :obj:`"AIDS700nef"`,
            :obj:`"LINUX"`, :obj:`"ALKANE"`, :obj:`"IMDBMulti"`).
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://drive.google.com/uc?export=download&id={}'

    datasets = {
        'AIDS700nef': {
            'id': '1Pl3CMFAR1MarzWUv7mjTtkDGoDo57-Ct',
            'extract': extract_zip,
            'pickle': '1fSWX05hLYvdOLv_-ayxf9MnaT720tWjz',
            'mcs_pickle': '13QvGzqoIdUctdowPWWtvfgP4AHsTD6px',
        },
        'LINUX': {
            'id': '1Ae9XUL1ST9OsnK7du8LLUpD2ZiZou9wQ',
            'extract': extract_tar,
            'pickle': '1TRPXryJevpZDbPCUTONrhP43Eb9noWzi',
            'mcs_pickle': '1ayp86EcMW5zPydssF_A6mBJ9h0hgUyjL',
        },
        'IMDBMulti': {
            'id': '1EZdhF5EAEDj1wqIBjA4Ds7YkTyVRJUxL',
            'extract': extract_zip,
            'pickle': '1HA7KztSbSVaS1ojiDZZqQ7iho6r0utmF',
            'mcs_pickle': '1V-DelKTvV84Nr60Y-sh27_BqChG2M9Lu',
        },
        'PTC': {
            'id': '1fDkW7EGnYO7nWIX2PcNNe8gNwlf8o_20',
            'id_mcs': '14CZNSV3t8qY-beHntK1OV8hb-9afhXsB',
            'extract': extract_zip,
            'pickle': '1hf8ZbxDKseMUEfDZgBaLzaFfA391vWsh',
            'mcs_pickle': '1MQAmNV4ABoOASbdY34uMkJr8R3Osy2Wo',
        },
    }

    # List of atoms contained in the AIDS700nef dataset:
    types = [
        'O', 'S', 'C', 'N', 'Cl', 'Br', 'B', 'Si', 'Hg', 'I', 'Bi', 'P', 'F',
        'Cu', 'Ho', 'Pd', 'Ru', 'Pt', 'Sn', 'Li', 'Ga', 'Tb', 'As', 'Co', 'Pb',
        'Sb', 'Se', 'Ni', 'Te'
    ]

    # List of atoms contained in the PTC dataset:
    types_ptc = [
        '2', '3', '4', '5', '6', '7', '8', '9', '10', '12', '14', '15', '16',
        '17', '18', '19', '20', '21', '22',
    ]

    def __init__(self,
                 args,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.args = args
        self.name = args.dataset
        assert self.name in self.datasets.keys()
        super().__init__("datasets/{}".format(args.dataset), transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
        path = osp.join(self.processed_dir, f'{self.name}_ged.pt')
        self.ged = torch.load(path)
        path = osp.join(self.processed_dir, f'{self.name}_norm_ged.pt')
        self.norm_ged = torch.load(path)
        path = osp.join(self.processed_dir, f'{self.name}_mcs.pt')
        self.mcs = torch.load(path)
        path = osp.join(self.processed_dir, f'{self.name}_norm_mcs.pt')
        self.norm_mcs = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        # Returns, e.g., ['LINUX/train', 'LINUX/test']
        return [osp.join(self.name, s) for s in ['train', 'test']]

    @property
    def processed_file_names(self) -> List[str]:
        # Returns, e.g., ['LINUX_training.pt', 'LINUX_test.pt']
        return [f'{self.name}_{s}.pt' for s in ['training', 'test']]

    def download(self):
        # Downloads the .tar/.zip file of the graphs and extracts them:
        if self.args.dataset == 'PTC' and self.args.metric == 'mcs':
            name = self.datasets[self.name]['id_mcs']
        else:
            name = self.datasets[self.name]['id']
        path = download_url(self.url.format(name), self.raw_dir)
        self.datasets[self.name]['extract'](path, self.raw_dir)
        os.unlink(path)

        # Downloads the pickle file containing pre-computed GEDs:
        name = self.datasets[self.name]['pickle']
        path = download_url(self.url.format(name), self.raw_dir)
        os.rename(path, osp.join(self.raw_dir, self.name, 'ged.pickle'))

        # Downloads the pickle file containing pre-computed MCSs:
        name = self.datasets[self.name]['mcs_pickle']
        path = download_url(self.url.format(name), self.raw_dir)
        os.rename(path, osp.join(self.raw_dir, self.name, 'mcs.pickle'))

    def process(self):
        import networkx as nx

        ids, Ns = [], []
        # Iterating over paths for raw and processed data (train + test):
        for k, (r_path, p_path) in enumerate(zip(self.raw_paths, self.processed_paths)):
            # Find the paths of all raw graphs:
            names = glob.glob(osp.join(r_path, '*.gexf'))
            # Get sorted graph IDs given filename: 123.gexf -> 123
            ids.append(sorted([int(i.split(os.sep)[-1][:-5]) for i in names]))

            data_list = []
            # Convert graphs in .gexf format to a NetworkX Graph:
            for i, idx in enumerate(ids[-1]):
                i = i if len(ids) == 1 else i + len(ids[0])
                # Reading the raw `*.gexf` graph:
                G = nx.read_gexf(osp.join(r_path, f'{idx}.gexf'))
                # Mapping of nodes in `G` to a contiguous number:
                mapping = {name: j for j, name in enumerate(G.nodes())}
                G = nx.relabel_nodes(G, mapping)
                Ns.append(G.number_of_nodes())

                edge_index = torch.tensor(list(G.edges)).t().contiguous()
                if edge_index.numel() == 0:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_index = to_undirected(edge_index, num_nodes=Ns[-1])
                data = Data(edge_index=edge_index, i=i)
                data.num_nodes = Ns[-1]

                # Create a one-hot encoded feature matrix denoting the atom
                # type (for the `AIDS700nef` and 'PTC' dataset):
                if self.name == 'AIDS700nef':
                    x = torch.zeros(data.num_nodes, dtype=torch.long)
                    for node, info in G.nodes(data=True):
                        x[int(node)] = self.types.index(info['type'])
                    data.x = F.one_hot(x, num_classes=len(self.types)).to(
                        torch.float)
                elif self.name == 'PTC':
                    x = torch.zeros(data.num_nodes, dtype=torch.long)
                    for node, info in G.nodes(data=True):
                        x[int(node)] = self.types_ptc.index(info['type'])
                    data.x = F.one_hot(x, num_classes=len(self.types_ptc)).to(
                        torch.float)

                # Benchmarking: positioning encoding.
                # data = positional_encoding(data, self.args.pe_dim)
                data = random_walk_positional_encoding(data, self.args.pe_dim)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            torch.save(self.collate(data_list), p_path)

        assoc = {idx: i for i, idx in enumerate(ids[0])}
        assoc.update({idx: i + len(ids[0]) for i, idx in enumerate(ids[1])})

        # Extracting ground-truth GEDs from the GED pickle file
        path = osp.join(self.raw_dir, self.name, 'ged.pickle')
        # Initialize GEDs as float('inf'):
        mat = torch.full((len(assoc), len(assoc)), float('inf'))
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            xs, ys, gs = [], [], []
            for (x, y), g in obj.items():
                xs += [assoc[x]]
                ys += [assoc[y]]
                gs += [g]
            # The pickle file does not contain GEDs for test graph pairs, i.e.
            # GEDs for (test_graph, test_graph) pairs are still float('inf'):
            x, y = torch.tensor(xs), torch.tensor(ys)
            ged = torch.tensor(gs, dtype=torch.float)
            mat[x, y], mat[y, x] = ged, ged

        path = osp.join(self.processed_dir, f'{self.name}_ged.pt')
        torch.save(mat, path)

        # Calculate the normalized GEDs:
        N = torch.tensor(Ns, dtype=torch.float)
        norm_mat = mat / (0.5 * (N.view(-1, 1) + N.view(1, -1)))

        path = osp.join(self.processed_dir, f'{self.name}_norm_ged.pt')
        torch.save(norm_mat, path)

        # Extracting ground-truth MCSs from the MCS pickle file
        path = osp.join(self.raw_dir, self.name, 'mcs.pickle')
        # Initialize MCSs as float(0):
        mat = torch.full((len(assoc), len(assoc)), float(0))
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            xs, ys, gs = [], [], []
            for (x, y), g in obj.items():
                xs += [assoc[x]]
                ys += [assoc[y]]
                gs += [g]
            # The pickle file does not contain MCSs for test graph pairs, i.e.
            # MCSs for (test_graph, test_graph) pairs are still float(0):
            x, y = torch.tensor(xs), torch.tensor(ys)
            mcs = torch.tensor(gs, dtype=torch.float)
            mat[x, y], mat[y, x] = mcs, mcs

        path = osp.join(self.processed_dir, f'{self.name}_mcs.pt')
        torch.save(mat, path)

        # Calculate the normalized MCSs:
        N = torch.tensor(Ns, dtype=torch.float)
        norm_mcs_mat = mat / (0.5 * (N.view(-1, 1) + N.view(1, -1)))

        path = osp.join(self.processed_dir, f'{self.name}_norm_mcs.pt')
        torch.save(norm_mcs_mat, path)

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'