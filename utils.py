import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
from args import args
import random


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_realdata(nu):
        E=[]
        labels=[]
        with open('{}nodes.txt'.format(nu), 'r') as f:
                list1 = f.readlines()
        f.close()
        num_edges = int(list1[0].split()[1])
        for i in range(num_edges):
                m, n = list(map(int, list1[i+1].split()))
                E.append([m, n])
                E.append([n, m])
        E1=np.array(E)    
        with open('label{}nodes.txt'.format(nu), 'r') as f:
                list1 = f.readlines()
        f.close()
        for i in range(len(list1)):
                labels.append(int(list1[i]))
        labels=torch.LongTensor(np.array(labels))
        onehot_label=torch.zeros(nu, 2).scatter_(1, labels.view(nu,1), 1)
        group=[[] for i in range(labels.max().item()+1)]
        
        for i in range(nu):
           for j in range(labels.max().item()+1):
               if labels[i]==j:
                  group[j].append(i)
                  break
       
        idx=[]
        for i in range(labels.max().item()+1):
           #group[i]=np.delete(group[i],a,0)
           random.shuffle(group[i])
           idx.append(group[i])
        idx_train=np.hstack([idx[i][0:10] for i in range(labels.max().item()+1)]) 
        idx_val=np.hstack([idx[i][10:20] for i in range(labels.max().item()+1)])
        idx_test=np.hstack([idx[i][20:-1] for i in range(labels.max().item()+1)])
        
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        return E1,labels,onehot_label,idx_train,idx_val,idx_test
def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
   
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # print(graph)
    edgelist=list(nx.from_dict_of_lists(graph).edges())
    print(len(edgelist))
    #print(edge_G.shape[0])
    #adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # adj=adj+adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) 
    #print(adj)
    # print(adj.shape[0])
    #print(adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    onehot_label=torch.FloatTensor(labels)
    # print(labels)
    labels_=labels.sum(1)
    
    #find isolated nodes
    a=[]
    for i in range(len(labels_)):
        if labels_[i]==0:
           a.append(i)
    for i in range(len(a)):
        labels[a[i]]=[0,0,0,0,0,1]
    print(len(a))
    print(a)
    a=np.array(a)
    labels = torch.LongTensor(np.where(labels)[1])
    if args.split==0:
       idx_test = test_idx_range.tolist() 
       idx_test=np.delete(idx_test,a,0)
       idx_train = range(len(y))
       idx_val = range(len(y), len(y)+500)
    else:
       group=[[] for i in range(labels.max().item()+1)]
       nu=labels.shape[0]
       for i in range(nu):
           for j in range(labels.max().item()+1):
               if labels[i]==j:
                  group[j].append(i)
                  break
       
       idx=[]
       for i in range(labels.max().item()+1):
           group[i]=np.delete(group[i],a,0)
           random.shuffle(group[i])
           idx.append(group[i])
       idx_train=np.hstack([idx[i][0:20] for i in range(labels.max().item()+1)]) 
       idx_val=np.hstack([idx[i][20:200] for i in range(labels.max().item()+1)])
       idx_test=np.hstack([idx[i][200:534] for i in range(labels.max().item()+1)])
       
      
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    
    E1=np.array(edgelist)
    
    E2,shape2=sparse_to_tuple(features)
    
    E1_=np.ones(E1.shape,dtype=int)
    E1_[:,0:1]=E1[:,1:2]
    E1_[:,1:2]=E1[:,0:1]
   
    E3=np.ones(E2.shape,dtype=int)
    E3[:,0:1]=E2[:,1:2]
    E3[:,1:2]=E2[:,0:1]
    #print(E2)
    #print(E3)
    #print(E1)
    print(idx_train)
    print(labels)
    
    #print(labels.shape[0])
    #print(shape2[1])
    #print(labels)
    return np.vstack((E1,E1_)), E2,E3, shape2,labels,onehot_label,idx_train, idx_val, idx_test
 

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        
        shape = mx.shape
        return coords,  shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] ,shape= to_tuple(sparse_mx[i])
    else:
        sparse_mx ,shape= to_tuple(sparse_mx)

    return sparse_mx,shape


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
if __name__=="__main__":
   dataset_str='citeseer'
   load_data(dataset_str)






