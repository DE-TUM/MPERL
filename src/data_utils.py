#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from src.utils import *
import time
import os, re, sys, gzip
import numpy as np
import scipy.sparse as sp
import rdflib as rdf
import glob
import pandas as pd
import wget
import pickle as pkl
import torch
from collections import Counter


class RDFReader:
    __graph = None
    __freq = {}

    def __init__(self, file):

        self.__graph = rdf.Graph()

        if file.endswith("nt.gz"):
            with gzip.open(file, "rb") as f:
                self.__graph.parse(file=f, format="nt")
        else:
            self.__graph.parse(file, format=rdf.util.guess_format(file))

        # See http://rdflib.readthedocs.io for the rdflib documentation

        self.__freq = Counter(self.__graph.predicates())

        print("Graph loaded, frequencies counted.")

    def triples(self, relation=None):
        for s, p, o in self.__graph.triples((None, relation, None)):
            yield s, p, o

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__graph.destroy("store")
        self.__graph.close(True)

    def subjectSet(self):
        return set(self.__graph.subjects())

    def objectSet(self):
        return set(self.__graph.objects())

    def relationList(self):
        """
        Returns a list of relations, ordered descending by frequenecy
        :return:
        """
        res = list(set(self.__graph.predicates()))
        res.sort(key=lambda rel: -self.freq(rel))
        return res

    def __len__(self):
        return len(self.__graph)

    def freq(self, relation):
        """zzz
        The frequency of this relation (how many distinct triples does it occur in?)
        :param relation:
        :return:
        """
        if relation not in self.__freq:
            return 0
        return self.__freq[relation]


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix(
        (loader["data"], loader["indices"], loader["indptr"]),
        shape=loader["shape"],
        dtype=np.float32,
    )


def save_sparse_csr(filename, array):
    np.savez(
        filename,
        data=array.data,
        indices=array.indices,
        indptr=array.indptr,
        shape=array.shape,
    )


def load_data(dataset_str="aifb", K=2, limit=-1):
    """
    :param dataset_str:
    :param rel_layers:
    :param limit: If > 0, will only load this many adj. matrices
        All adjacencies are preloaded and saved to disk,
        but only a limited a then restored to memory.
    :return:
    """

    print("Loading dataset", dataset_str)

    # dirname = os.path.dirname(os.path.realpath(sys.argv[0]))
    dirname = os.getcwd() + "/"

    if dataset_str == "aifb":
        data_url = "https://www.dropbox.com/s/fkvgvkygo2gf28k/aifb_stripped.nt.gz?dl=1"
        # The RDF file containing the knowledge graph
        graph_file = "data/aifb/aifb_stripped.nt.gz"
        if not os.path.isfile(graph_file):
            print("Downloading AIFB data.")
            wget.download(data_url, graph_file)

        # The TSV file containing the classification task
        task_file = "data/aifb/completeDataset.tsv"
        # The TSV file containing training indices
        train_file = "data/aifb/trainingSet.tsv"
        # The TSV file containing test indices
        test_file = "data/aifb/testSet.tsv"
        label_header = "label_affiliation"
        nodes_header = "person"

    elif dataset_str == "mutag":
        data_url = "https://www.dropbox.com/s/qy8j3p8eacvm4ir/mutag_stripped.nt.gz?dl=1"
        graph_file = "data/mutag/mutag_stripped.nt.gz"
        if not os.path.isfile(graph_file):
            print("Downloading MUTAG data.")
            wget.download(data_url, graph_file)
        task_file = "data/mutag/completeDataset.tsv"
        train_file = "data/mutag/trainingSet.tsv"
        test_file = "data/mutag/testSet.tsv"
        label_header = "label_mutagenic"
        nodes_header = "bond"

    elif dataset_str == "bgs":
        data_url = "https://www.dropbox.com/s/uqi0k9jd56j02gh/bgs_stripped.nt.gz?dl=1"
        graph_file = "data/bgs/bgs_stripped.nt.gz"
        if not os.path.isfile(graph_file):
            print("Downloading BGS data.")
            wget.download(data_url, graph_file)
        task_file = "data/bgs/completeDataset_lith.tsv"
        train_file = "data/bgs/trainingSet(lith).tsv"
        test_file = "data/bgs/testSet(lith).tsv"
        label_header = "label_lithogenesis"
        nodes_header = "rock"
    elif dataset_str == "am":
        data_url = "https://www.dropbox.com/s/htisydfgwxmrx65/am_stripped.nt.gz?dl=1"
        graph_file = "data/am/am_stripped.nt.gz"

        if not os.path.isfile(graph_file):
            print("Downloading AM data.")
            wget.download(data_url, graph_file)

        task_file = "data/am/completeDataset.tsv"
        train_file = "data/am/trainingSet.tsv"
        test_file = "data/am/testSet.tsv"
        label_header = "label_cateogory"
        nodes_header = "proxy"

    else:
        raise NameError("Dataset name not recognized: " + dataset_str)

    adj_fprepend = "data/" + dataset_str + "/adjacencies_"
    labels_file = "data/" + dataset_str + "/labels.npz"
    train_idx_file = "data/" + dataset_str + "/train_idx.npy"
    test_idx_file = "data/" + dataset_str + "/test_idx.npy"
    train_names_file = "data/" + dataset_str + "/train_names.npy"
    test_names_file = "data/" + dataset_str + "/test_names.npy"
    rel_dict_file = "data/" + dataset_str + "/rel_dict.pkl"
    nodes_file = "data/" + dataset_str + "/nodes.pkl"

    graph_file = dirname + "/" + graph_file
    task_file = dirname + "/" + task_file
    train_file = dirname + "/" + train_file
    test_file = dirname + "/" + test_file
    adj_fprepend = dirname + "/" + adj_fprepend
    labels_file = dirname + "/" + labels_file
    train_idx_file = dirname + "/" + train_idx_file
    test_idx_file = dirname + "/" + test_idx_file
    train_names_file = dirname + "/" + train_names_file
    test_names_file = dirname + "/" + test_names_file
    rel_dict_file = dirname + "/" + rel_dict_file
    nodes_file = dirname + "/" + nodes_file

    adj_files = glob.glob(adj_fprepend + "*.npz")
    if (
        adj_files != []
        and os.path.isfile(labels_file)
        and os.path.isfile(train_idx_file)
        and os.path.isfile(test_idx_file)
    ):

        # load precomputed adjacency matrix and labels

        adj_files.sort(
            key=lambda f: int(re.search("adjacencies_(.+?).npz", f).group(1))
        )

        if limit > 0:
            adj_files = adj_files[: limit * 2]

        adjacencies = [load_sparse_csr(file) for file in adj_files]
        adj_shape = adjacencies[0].shape

        print("Number of nodes: ", adj_shape[0])
        print("Number of relations: ", len(adjacencies))

        labels = load_sparse_csr(labels_file)
        labeled_nodes_idx = list(labels.nonzero()[0])

        print("Number of classes: ", labels.shape[1])

        train_idx = np.load(train_idx_file)
        test_idx = np.load(test_idx_file)
        train_names = np.load(train_names_file)
        test_names = np.load(test_names_file)

        relations_dict = pkl.load(open(rel_dict_file, "rb"))

    else:

        # loading labels of nodes
        labels_df = pd.read_csv(task_file, sep="\t", encoding="utf-8")
        labels_train_df = pd.read_csv(train_file, sep="\t", encoding="utf8")
        labels_test_df = pd.read_csv(test_file, sep="\t", encoding="utf8")
        with RDFReader(graph_file) as reader:

            relations = reader.relationList()
            subjects = reader.subjectSet()
            objects = reader.objectSet()

            print([(rel, reader.freq(rel)) for rel in relations[:limit]])

            nodes = list(subjects.union(objects))
            # fixed order
            # nodes.sort()
            adj_shape = (len(nodes), len(nodes))

            print("Number of nodes: ", len(nodes))
            print("Number of relations in the data: ", len(relations))

            relations_dict = {rel: i for i, rel in enumerate(list(relations))}
            nodes_dict = {node: i for i, node in enumerate(nodes)}
            assert len(nodes_dict) < np.iinfo(np.int32).max

            adjacencies = []

            for i, rel in enumerate(relations if limit < 0 else relations[:limit]):

                print(
                    u"Creating adjacency matrix for relation {}: {}, frequency {}".format(
                        i, rel, reader.freq(rel)
                    )
                )
                edges = np.empty((reader.freq(rel), 2), dtype=np.int32)

                size = 0
                for j, (s, p, o) in enumerate(reader.triples(relation=rel)):
                    if nodes_dict[s] > len(nodes) or nodes_dict[o] > len(nodes):
                        print(s, o, nodes_dict[s], nodes_dict[o])

                    edges[j] = np.array([nodes_dict[s], nodes_dict[o]])
                    size += 1

                print("{} edges added".format(size))

                row, col = np.transpose(edges)

                data = np.ones(len(row), dtype=np.int8)

                adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.int8)

                adj_transp = sp.csr_matrix(
                    (data, (col, row)), shape=adj_shape, dtype=np.int8
                )

                save_sparse_csr(adj_fprepend + "%d.npz" % (i * 2), adj)
                save_sparse_csr(adj_fprepend + "%d.npz" % (i * 2 + 1), adj_transp)

                if limit < 0:
                    adjacencies.append(adj)
                    adjacencies.append(adj_transp)

        # Reload the adjacency matrices from disk
        if limit > 0:
            adj_files = glob.glob(adj_fprepend + "*.npz")
            adj_files.sort(
                key=lambda f: int(re.search("adjacencies_(.+?).npz", f).group(1))
            )

            adj_files = adj_files[: limit * 2]
            for i, file in enumerate(adj_files):
                adjacencies.append(load_sparse_csr(file))
                print("%d adjacency matrices loaded " % i)
        nodes_u_dict = {
            np.unicode(to_unicode(key)): val for key, val in nodes_dict.items()
        }

        labels_set = set(labels_df[label_header].values.tolist())
        labels_dict = {lab: i for i, lab in enumerate(list(labels_set))}
        print("{} classes: {}", len(labels_set), labels_set)

        labels = sp.lil_matrix((adj_shape[0], len(labels_set)))
        labeled_nodes_idx = []

        print("Loading training set")

        train_idx = []
        train_names = []
        for nod, lab in zip(
            labels_train_df[nodes_header].values, labels_train_df[label_header].values
        ):
            nod = np.unicode(to_unicode(nod))  # type: unicode
            if nod in nodes_u_dict:
                labeled_nodes_idx.append(nodes_u_dict[nod])
                label_idx = labels_dict[lab]
                labels[labeled_nodes_idx[-1], label_idx] = 1
                train_idx.append(nodes_u_dict[nod])
                train_names.append(nod)
            else:
                print(
                    u"Node not in dictionary, skipped: ",
                    nod.encode("utf-8", errors="replace"),
                )

        print("Loading test set")
        test_idx = []
        test_names = []
        for nod, lab in zip(
            labels_test_df[nodes_header].values, labels_test_df[label_header].values
        ):
            nod = np.unicode(to_unicode(nod))
            if nod in nodes_u_dict:
                labeled_nodes_idx.append(nodes_u_dict[nod])
                label_idx = labels_dict[lab]
                labels[labeled_nodes_idx[-1], label_idx] = 1
                test_idx.append(nodes_u_dict[nod])
                test_names.append(nod)
            else:
                print(
                    u"Node not in dictionary, skipped: ",
                    nod.encode("utf-8", errors="replace"),
                )

        labeled_nodes_idx = sorted(labeled_nodes_idx)
        labels = labels.tocsr()

        save_sparse_csr(labels_file, labels)

        np.save(train_idx_file, train_idx)
        np.save(test_idx_file, test_idx)

        np.save(train_names_file, train_names)
        np.save(test_names_file, test_names)

        pkl.dump(relations_dict, open(rel_dict_file, "wb"))
        pkl.dump(nodes, open(nodes_file, "wb"))

        # add identity matrix (self-connections)
        adjacencies.append(sp.identity(len(nodes)).tocsr())

        # Delete unused nodes for memory efficiency
        bfs_generator = bfs_relational(adjacencies, labeled_nodes_idx)
        lvls = set(labeled_nodes_idx)
        for hop in range(K):
            lvls |= set.union(*bfs_generator.__next__())  # K-hop neighbors
        # R-GCN
        todel = list(set(range(len(nodes))) - lvls)
        for i in range(len(adjacencies)):
            csr_zero_rows(adjacencies[i], todel)

    features = sp.identity(adj_shape[0], format="csr")
    return (
        adjacencies,
        features,
        labels,
        labeled_nodes_idx,
        train_idx,
        test_idx,
        relations_dict,
        train_names,
        test_names,
    )


def parse(symbol):
    if symbol.startswith("<"):
        return symbol[1:-1]
    return symbol


def to_unicode(input):
    if isinstance(input, str):
        return input
    elif isinstance(input, bytes):
        return input.decode("utf-8", errors="replace")
    return str(input).decode("utf-8", errors="replace")


# load data especially for YAGO and FB datasets
def load_data_yago_fb(dataset_str="YAGO43k", K=2, limit=-1):
    """
    :param dataset_str:
    :param rel_layers:
    :param limit: If > 0, will only load this many adj. matrices
        All adjacencies are preloaded and saved to disk,
        but only a limited a then restored to memory.
    """

    print("Loading dataset", dataset_str)

    # dirname = os.path.dirname(os.path.realpath(sys.argv[0]))
    dirname = os.getcwd() + "/"

    if dataset_str == "YAGO43k":
        data_url = "https://www.dropbox.com/scl/fi/w6qeke18c91sox7ubprxe/YAGO43k.ttl?rlkey=6x8d4brnp6tunyfi761i1al5o&dl=1"
        graph_file = "data/YAGO43k/YAGO43k.ttl"

        if not os.path.isfile(graph_file):
            print("Downloading YAGO43k data.")
            wget.download(data_url, graph_file)
        
        task_file = "data/YAGO43k/completeDataset.tsv"
        train_file = "data/YAGO43k/trainingSet.tsv"
        valid_file = "data/YAGO43k/validSet.tsv"
        test_file = "data/YAGO43k/testSet.tsv"
        label_header = "label_class"
        nodes_header = "entity"
    
    elif dataset_str == "FB15k":
        data_url = "https://www.dropbox.com/scl/fi/uy131jxma692jzh6u7930/FB15k.nt.gz?rlkey=od1uvgz5kapwkri8hyvldldqx&dl=1"
        graph_file = "data/FB15k/FB15k.nt.gz"

        if not os.path.isfile(graph_file):
            print("Downloading FB15k data.")
            wget.download(data_url, graph_file)
        
        task_file = "data/FB15k/completeDataset.tsv"
        train_file = "data/FB15k/trainingSet.tsv"
        valid_file = "data/FB15k/validSet.tsv"
        test_file = "data/FB15k/testSet.tsv"
        label_header = "label_class"
        nodes_header = "entity"

    else:
        raise NameError("Dataset name not recognized: " + dataset_str)

    adj_fprepend = "data/" + dataset_str + "/adjacencies_"
    labels_file = "data/" + dataset_str + "/labels.npz" # contains all_true labels for each entity
    train_labels_file = "data/" + dataset_str + "/train_labels.npz" # contains only train labels for each entity
    valid_labels_file = "data/" + dataset_str + "/valid_labels.npz"
    test_labels_file = "data/" + dataset_str + "/test_labels.npz"
    train_idx_file = "data/" + dataset_str + "/train_idx.npy"
    valid_idx_file = "data/" + dataset_str + "/valid_idx.npy"
    test_idx_file = "data/" + dataset_str + "/test_idx.npy"
    train_names_file = "data/" + dataset_str + "/train_names.npy"
    valid_names_file = "data/" + dataset_str + "/valid_names.npy"
    test_names_file = "data/" + dataset_str + "/test_names.npy"
    rel_dict_file = "data/" + dataset_str + "/rel_dict.pkl"
    nodes_file = "data/" + dataset_str + "/nodes.pkl"

    graph_file = dirname + "/" + graph_file
    task_file = dirname + "/" + task_file
    train_file = dirname + "/" + train_file
    valid_file = dirname + "/" + valid_file
    test_file = dirname + "/" + test_file
    adj_fprepend = dirname + "/" + adj_fprepend
    labels_file = dirname + "/" + labels_file
    train_labels_file = dirname + "/" + train_labels_file
    valid_labels_file = dirname + "/" + valid_labels_file
    test_labels_file = dirname + "/" + test_labels_file
    train_idx_file = dirname + "/" + train_idx_file
    valid_idx_file = dirname + "/" + valid_idx_file
    test_idx_file = dirname + "/" + test_idx_file
    train_names_file = dirname + "/" + train_names_file
    valid_names_file = dirname + "/" + valid_names_file
    test_names_file = dirname + "/" + test_names_file
    rel_dict_file = dirname + "/" + rel_dict_file
    nodes_file = dirname + "/" + nodes_file

    adj_files = glob.glob(adj_fprepend + "*.npz")
    if (
        adj_files != []
        and os.path.isfile(labels_file)
        and os.path.isfile(train_labels_file)
        and os.path.isfile(valid_labels_file)
        and os.path.isfile(test_labels_file)
        and os.path.isfile(train_idx_file)
        and os.path.isfile(valid_idx_file)
        and os.path.isfile(test_idx_file)
    ):

        # load precomputed adjacency matrix and labels

        adj_files.sort(
            key=lambda f: int(re.search("adjacencies_(.+?).npz", f).group(1))
        )

        if limit > 0:
            adj_files = adj_files[: limit * 2]

        adjacencies = [load_sparse_csr(file) for file in adj_files]
        adj_shape = adjacencies[0].shape

        print("Number of nodes: ", adj_shape[0])
        print("Number of relations: ", len(adjacencies))

        labels = load_sparse_csr(labels_file)
        train_labels = load_sparse_csr(train_labels_file)
        valid_labels = load_sparse_csr(valid_labels_file)
        test_labels = load_sparse_csr(test_labels_file)
        labeled_nodes_idx = list(set(labels.nonzero()[0])) # labeled_nodes

        print("Number of classes: ", labels.shape[1])

        train_idx = np.load(train_idx_file)
        test_idx = np.load(test_idx_file)
        valid_idx = np.load(valid_idx_file)
        train_names = np.load(train_names_file)
        test_names = np.load(test_names_file)
        valid_names = np.load(valid_names_file)

        relations_dict = pkl.load(open(rel_dict_file, "rb"))

    else:

        # loading labels of nodes
        labels_df = pd.read_csv(task_file, sep="\t", encoding="utf-8") # all labels
        labels_train_df = pd.read_csv(train_file, sep="\t", encoding="utf8")
        labels_valid_df = pd.read_csv(valid_file, sep="\t", encoding="utf8")
        labels_test_df = pd.read_csv(test_file, sep="\t", encoding="utf8")

        with RDFReader(graph_file) as reader:
            relations = reader.relationList()
            subjects = reader.subjectSet()
            objects = reader.objectSet()

            print([(rel, reader.freq(rel)) for rel in relations[:limit]])

            nodes = list(subjects.union(objects))
            # fixed order
            # nodes.sort()
            adj_shape = (len(nodes), len(nodes))

            print("Number of nodes: ", len(nodes))
            print("Number of relations in the data: ", len(relations))

            relations_dict = {rel: i for i, rel in enumerate(list(relations))}
            nodes_dict = {node: i for i, node in enumerate(nodes)}
            assert len(nodes_dict) < np.iinfo(np.int32).max

            adjacencies = []
            for i, rel in enumerate(relations if limit < 0 else relations[:limit]):

                print(
                    u"Creating adjacency matrix for relation {}: {}, frequency {}".format(
                        i, rel, reader.freq(rel)
                    )
                )
                edges = np.empty((reader.freq(rel), 2), dtype=np.int32)

                size = 0
                for j, (s, p, o) in enumerate(reader.triples(relation=rel)):
                    if nodes_dict[s] > len(nodes) or nodes_dict[o] > len(nodes):
                        print(s, o, nodes_dict[s], nodes_dict[o])

                    edges[j] = np.array([nodes_dict[s], nodes_dict[o]])
                    size += 1

                print("{} edges added".format(size))

                row, col = np.transpose(edges)

                data = np.ones(len(row), dtype=np.int8)

                adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.int8)

                adj_transp = sp.csr_matrix(
                    (data, (col, row)), shape=adj_shape, dtype=np.int8
                )

                save_sparse_csr(adj_fprepend + "%d.npz" % (i * 2), adj)
                save_sparse_csr(adj_fprepend + "%d.npz" % (i * 2 + 1), adj_transp)

                if limit < 0:
                    # add edges for both directions
                    adjacencies.append(adj)
                    adjacencies.append(adj_transp)

        # Reload the adjacency matrices from disk
        if limit > 0:
            adj_files = glob.glob(adj_fprepend + "*.npz")
            adj_files.sort(
                key=lambda f: int(re.search("adjacencies_(.+?).npz", f).group(1))
            )

            adj_files = adj_files[: limit * 2]
            for i, file in enumerate(adj_files):
                adjacencies.append(load_sparse_csr(file))
                print("%d adjacency matrices loaded " % i)
        
        nodes_u_dict = {
            np.unicode(to_unicode(key)): val for key, val in nodes_dict.items()
        }
        id2nodes_u = {val: np.unicode(to_unicode(key)) for key, val in nodes_dict.items()}

        labels_set = set(labels_df[label_header].values.tolist())
        labels_dict = {lab: i for i, lab in enumerate(list(labels_set))}
        print("{} classes: {}", len(labels_set), labels_set)

        labels = sp.lil_matrix((adj_shape[0], len(labels_set))) # all_true, shape = (n_entities, n_classes)
        labeled_nodes_idx = []

        print("Loading training set")
        train_labels = sp.lil_matrix((adj_shape[0], len(labels_set)))
        for nod, lab in zip(
            labels_train_df[nodes_header].values, labels_train_df[label_header].values
        ):
            nod = np.unicode(to_unicode(nod))  # type: unicode
            if nod in nodes_u_dict:
                labeled_nodes_idx.append(nodes_u_dict[nod])
                node_idx = nodes_u_dict[nod]
                label_idx = labels_dict[lab]
                labels[node_idx, label_idx] = 1 # all
                train_labels[node_idx, label_idx] = 1 # only train
                # train_idx.append(nodes_u_dict[nod])
                # train_names.append(nod)
            else:
                print(
                    u"Node not in dictionary, skipped: ",
                    nod.encode("utf-8", errors="replace"),
                )
        train_idx = train_labels.sum(axis=1).nonzero()[0] # array
        train_names = [id2nodes_u[i] for i in train_idx] # list


        print("Loading valid set")
        valid_labels = sp.lil_matrix((adj_shape[0], len(labels_set)))
        for nod, lab in zip(
            labels_valid_df[nodes_header].values, labels_valid_df[label_header].values
        ):
            nod = np.unicode(to_unicode(nod))
            if nod in nodes_u_dict:
                labeled_nodes_idx.append(nodes_u_dict[nod])
                node_idx = nodes_u_dict[nod]
                label_idx = labels_dict[lab]
                labels[node_idx, label_idx] = 1
                valid_labels[node_idx, label_idx] = 1
            else:
                print(
                    u"Node not in dictionary, skipped: ",
                    nod.encode("utf-8", errors="replace"),
                )
        valid_idx = valid_labels.sum(axis=1).nonzero()[0]
        valid_names = [id2nodes_u[i] for i in valid_idx]


        print("Loading test set")
        test_labels = sp.lil_matrix((adj_shape[0], len(labels_set)))
        for nod, lab in zip(
            labels_test_df[nodes_header].values, labels_test_df[label_header].values
        ):
            nod = np.unicode(to_unicode(nod))
            if nod in nodes_u_dict:
                labeled_nodes_idx.append(nodes_u_dict[nod])
                node_idx = nodes_u_dict[nod]
                label_idx = labels_dict[lab]
                labels[node_idx, label_idx] = 1
                test_labels[node_idx, label_idx] = 1
            else:
                print(
                    u"Node not in dictionary, skipped: ",
                    nod.encode("utf-8", errors="replace"),
                )
        test_idx = test_labels.sum(axis=1).nonzero()[0]
        test_names = [id2nodes_u[i] for i in test_idx]

        labeled_nodes_idx = sorted(list(set(labeled_nodes_idx)))
        labels = labels.tocsr()
        train_labels = train_labels.tocsr()
        valid_labels = valid_labels.tocsr()
        test_labels = test_labels.tocsr()

        save_sparse_csr(labels_file, labels)
        save_sparse_csr(train_labels_file, train_labels)
        save_sparse_csr(valid_labels_file, valid_labels)
        save_sparse_csr(test_labels_file, test_labels)

        np.save(train_idx_file, train_idx)
        np.save(valid_idx_file, valid_idx)
        np.save(test_idx_file, test_idx)

        np.save(train_names_file, train_names)
        np.save(valid_names_file, valid_names)
        np.save(test_names_file, test_names)

        pkl.dump(relations_dict, open(rel_dict_file, "wb"))
        pkl.dump(nodes, open(nodes_file, "wb"))

        # add identity matrix (self-connections)
        adjacencies.append(sp.identity(len(nodes)).tocsr())

        # Delete unused nodes for memory efficiency
        bfs_generator = bfs_relational(adjacencies, labeled_nodes_idx)
        lvls = set(labeled_nodes_idx)
        for hop in range(K):
            lvls |= set.union(*bfs_generator.__next__())  # K-hop neighbors
        # R-GCN
        todel = list(set(range(len(nodes))) - lvls)
        for i in range(len(adjacencies)):
            csr_zero_rows(adjacencies[i], todel)

    features = sp.identity(adj_shape[0], format="csr")
    return (
        adjacencies,
        features,
        labels,
        train_idx, # list containing indices of train nodes
        valid_idx,
        test_idx,
        relations_dict,
        train_names,
        valid_names,
        test_names,
        train_labels, # shape(n_entities, n_classes)
        valid_labels,
        test_labels,
    )