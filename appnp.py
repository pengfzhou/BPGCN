import math
import torch
from scipy import sparse
import numpy as np
from gatlayers import  SpGraphAttentionLayer
import torch.nn.functional as F

def normalize_adjacency_matrix(A, I):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde = A +I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sparse.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    
   # sparse_mx = sparse_mx.astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def create_propagator_matrix(A, alpha, model):
    """
    Creating  apropagation matrix.
    :param graph: NetworkX graph.
    :param alpha: Teleport parameter.
    :param model: Type of model exact or approximate.
    :return propagator: Propagator matrix - Dense torch matrix or dict with indices and values for sparse multiplication.
    """
    
    I = sparse.eye(A.shape[0])
    A_tilde_hat = normalize_adjacency_matrix(A, I)
    if model == "exact":
        propagator = (I-(1-alpha)*A_tilde_hat).todense()
        propagator = alpha*torch.inverse(torch.FloatTensor(propagator))
    else:
        propagator = dict()
        A_tilde_hat = sparse.coo_matrix(A_tilde_hat)
        propagator["indices"] = torch.LongTensor(np.concatenate([A_tilde_hat.row.reshape(-1,1), A_tilde_hat.col.reshape(-1,1)],axis=1).T)
        propagator["values"] = torch.FloatTensor(A_tilde_hat.data)
    return propagator
def uniform(size, tensor):
    """
    Uniform weight initialization.
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

class DenseFullyConnected(torch.nn.Module):
    """
    Abstract class for PageRank and Approximate PageRank networks.
    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param density: Feature matrix structure.
    """
    def __init__(self, in_channels, out_channels):
        super(DenseFullyConnected, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        uniform(self.out_channels, self.bias)

    def forward(self, features):
        """
        Doing a forward pass.
        :param features: Feature matrix.
        :return filtered_features: Convolved features.
        """
        filtered_features = torch.mm(features, self.weight_matrix)
        filtered_features = filtered_features + self.bias
        return filtered_features

class SparseFullyConnected(torch.nn.Module):
    """
    Abstract class for PageRank and Approximate PageRank networks.
    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param density: Feature matrix structure.
    """
    def __init__(self, in_channels, out_channels):
        super(SparseFullyConnected, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        uniform(self.out_channels, self.bias)

    def forward(self, feature):
        """
        Making a forward pass.
        :param feature_indices: Non zero value indices.
        :param feature_values: Matrix values.
        :return filtered_features: Output features.
        """
        
        filtered_features = torch.spmm(feature,self.weight_matrix)
        filtered_features = filtered_features + self.bias
        return filtered_features        
class GCN(torch.nn.Module):
    """
    APPNP Model Class.
    :param args: Arguments object.
    :param number_of_labels: Number of target labels.
    :param number_of_features: Number of input features.
    :param graph: NetworkX graph.
    :param device: CPU or GPU.
    """
    def __init__(self, args, number_of_labels, number_of_features, A) :
        super(GCN, self).__init__()
        self.args = args
        self.number_of_labels = number_of_labels
        self.number_of_features = number_of_features
        self.A=A
        self.Aindices=torch.from_numpy(np.vstack((A.row,A.col)).astype(np.int64))
        self.Avalues=torch.from_numpy(A.data)
        self.Ashape=torch.Size(A.shape)
        print(self.A)
        self.device = args.device
        self.setup_layers()
        self.setup_propagator()
        
    def setup_layers(self):
        """
        Creating layers.
        """
        self.layer_1 = SparseFullyConnected(self.number_of_features, self.args.hidden)
        self.layer_2 = DenseFullyConnected(self.args.hidden, self.number_of_labels)

    def setup_propagator(self):
        """
        Defining propagation matrix (Personalized Pagrerank or adjacency).
        """
        self.propagator = create_propagator_matrix(self.A, self.args.alpha, self.args.model)
        if self.args.model=="exact":
            self.propagator = self.propagator.to(self.device)
        else:
            self.edge_indices = self.propagator["indices"].to(self.device)
            self.edge_weights = self.propagator["values"].to(self.device)

    def forward(self, feature_indices, feature_values):
        """
        Making a forward propagation pass.
        :param feature_indices: Feature indices for feature matrix.
        :param feature_values: Values in the feature matrix.
        :return self.predictions: Predicted class label log softmaxes.
        """
        self.C=torch.sparse.FloatTensor(self.Aindices,self.Avalues,self.Ashape).to(self.device).float()
        
        self.A_=torch.sparse.FloatTensor(self.edge_indices,self.edge_weights,torch.Size([self.args.nu,self.args.nu])).to(self.device)
        feature=torch.sparse.FloatTensor(feature_indices,feature_values,torch.Size([self.args.nu,self.args.ne])).to(self.device)
        
        latent_features_1=  self.layer_1(feature)
        latent_features_1 = torch.nn.functional.relu(torch.spmm(self.A_,latent_features_1))
        latent_features_1 = torch.nn.functional.dropout(latent_features_1, p = self.args.dropout, training = self.training)
        latent_features_2 = self.layer_2(latent_features_1)
        localized_predictions=torch.spmm(self.A_,latent_features_2)

         
            
        self.predictions = torch.nn.functional.log_softmax(localized_predictions , dim=1)
        return self.predictions
class MLP(torch.nn.Module):
    """
    APPNP Model Class.
    :param args: Arguments object.
    :param number_of_labels: Number of ZZtarget labels.
    :param number_of_features: Number of input features.
    :param graph: NetworkX graph.
    :param device: CPU or GPU.
    """
    def __init__(self, args, number_of_labels, number_of_features, A) :
        super(MLP, self).__init__()
        self.args = args
        self.number_of_labels = number_of_labels
        self.number_of_features = number_of_features
        self.A=A
        self.device = args.device
        self.setup_layers()
        self.setup_propagator()

    def setup_layers(self):
        """ZZ
        Creating layers.
        """
        self.layer_1 = DenseFullyConnected(self.args.ne, self.args.hidden)
        self.layer_2 = DenseFullyConnected(self.args.hidden, self.number_of_labels)

    def setup_propagator(self):
        """
        Defining propagation matrix (Personalized Pagrerank or adjacency).
        """
        self.propagator = create_propagator_matrix(self.A, self.args.alpha, self.args.model)
        if self.args.model=="exact":
            self.propagator = self.propagator.to(self.device)
        else:
            self.edge_indices = self.propagator["indices"].to(self.device)
            self.edge_weights = self.propagator["values"].to(self.device)

    def forward(self, feature_indices, feature_values):
        """
        Making a forward propagation pass.
        :param feature_indices: Feature indices for feature matrix.
        :param feature_values: Values in the feature matrix.
        :return self.predictions: Predicted class label log softmaxes.
        """
        self.A_=torch.sparse.FloatTensor(self.edge_indices,self.edge_weights,torch.Size([self.args.nu,self.args.nu])).to(self.device)
        feature=torch.sparse.FloatTensor(feature_indices,feature_values,torch.Size([self.args.nu,self.args.ne])).to(self.device).to_dense()
        #B=feature @ feature.t()
        #B=B-torch.diagflat(torch.diagonal(B)) 
        #B=B+torch.eye(B.shape[0]).to(self.device)
       # B=torch.max(B-5,torch.zeros(B.shape[0],B.shape[0]).to(self.device))
        #print(B.max())
       # B=B/B.max()
       # B=torch.ceil(B/20.0)
       # B=torch.clamp(B-10,0,1)
       # np.savetxt("adja.txt",torch.sum(B[:,0:3999],1).cpu().numpy(),fmt="%1.0f")
        #degree=torch.sum(B,1)
       # print(degree)        
        #D=torch.sqrt(torch.diagflat(1.0/degree))
       # B=D @ B @ D
        
        latent_features_1 =  self.layer_1(feature)
        latent_features_1 = torch.nn.functional.relu(latent_features_1)
        latent_features_1 = torch.nn.functional.dropout(latent_features_1, p = self.args.dropout, training = self.training)
        latent_features_2 = self.layer_2(latent_features_1)
        #latent_features_2=torch.spmm(self.A_,latent_features_2)
        #for i in range(50):
           # latent_features_2=torch.spmm(self.A_,latent_features_2)
        localized_predictions=latent_features_2
         
            
        self.predictions = torch.nn.functional.log_softmax(localized_predictions , dim=1)
        return self.predictions
class APPNPModel(torch.nn.Module):
    """
    APPNP Model Class.
    :param args: Arguments object.
    :param number_of_labels: Number of target labels.
    :param number_of_features: Number of input features.
    :param graph: NetworkX graph.
    :param device: CPU or GPU.
    """
    def __init__(self, args, number_of_labels, number_of_features, A) :
        super(APPNPModel, self).__init__()
        self.args = args
        self.number_of_labels = number_of_labels
        self.number_of_features = number_of_features
        self.A=A
        self.device = args.device
        self.setup_layers()
        self.setup_propagator()

    def setup_layers(self):
        """
        Creating layers.
        """
        self.layer_1 = SparseFullyConnected(self.number_of_features, self.args.hidden)
        self.layer_2 = DenseFullyConnected(self.args.hidden, self.number_of_labels)

    def setup_propagator(self):
        """
        Defining propagation matrix (Personalized Pagrerank or adjacency).ZZ
        """
        self.propagator = create_propagator_matrix(self.A, self.args.alpha, self.args.model)
        if self.args.model=="exact":
            self.propagator = self.propagator.to(self.device)
        else:
            self.edge_indices = self.propagator["indices"].to(self.device)
            self.edge_weights = self.propagator["values"].to(self.device)

    def forward(self, feature_indices, feature_values):
        """
        Making a forward propagation pass.
        :param feature_indices: Feature indices for feature matrix.
        :param feature_values: Values in the feature matrix.
        :return self.predictions: Predicted class label log softmaxes.
        """
        
        
        #feature_values = torch.nn.functional.dropout(feature_values, p = self.args.dropout, training = self.training)
        feature=torch.sparse.FloatTensor(feature_indices,feature_values,torch.Size([self.args.nu,self.args.ne])).to(self.device).float()
        latent_features_1 = torch.nn.functional.relu(self.layer_1(feature))
        latent_features_1 = torch.nn.functional.dropout(latent_features_1, p = self.args.dropout, training = self.training)
        latent_features_2 = self.layer_2(latent_features_1)

        if self.args.model=="exact":       
            self.predictions = torch.mm(torch.nn.functional.dropout(self.propagator, p = self.args.dropout, training = self.training), latent_features_2)
        else:
            localized_predictions = latent_features_2
            edge_weights = torch.nn.functional.dropout(self.edge_weights, p = self.args.dropout, training = self.training)
            self.A_=torch.sparse.FloatTensor(self.edge_indices,self.edge_weights,torch.Size([self.args.nu,self.args.nu])).to(self.device).float()
            for iteration in range(self.args.iter_time):       
                localized_predictions = (1-self.args.alpha)*torch.spmm(self.A_, localized_predictions)+self.args.alpha*latent_features_2
            self.predictions = localized_predictions  
            
        self.predictions = torch.nn.functional.log_softmax(self.predictions , dim=1)
        return self.predictions
class SpGAT(torch.nn.Module):
    def __init__(self, args, number_of_labels, number_of_features,adj):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.args=args
        
        self.number_of_labels = number_of_labels
        self.number_of_features = number_of_features
        self.device = args.device
        self.adj= sparse_mx_to_torch_sparse_tensor(adj).to(self.device).to_dense()
        self.attentions = [SpGraphAttentionLayer(number_of_features, 
                                                 args.hidden, 
                                                 dropout=args.dropout, 
                                                 alpha=args.alpha, 
                                                 concat=True) for _ in range(args.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(args.hidden * args.nheads, 
                                             args.Q, 
                                             dropout=args.dropout, 
                                             alpha=args.alpha, 
                                             concat=False)
    
    def forward(self, feature_indices, feature_values):
        x=torch.sparse.FloatTensor(feature_indices,feature_values,torch.Size([self.args.nu,self.args.ne])).to(self.device).to_dense().float()
        x = F.dropout(x, self.args.dropout, training=self.training)
        x = torch.cat([att(x, self.adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.args.dropout, training=self.training)
        x = F.elu(self.out_att(x, self.adj))
        return F.log_softmax(x, dim=1)
class SGCN(torch.nn.Module):
    def __init__(self,args,number_of_labels,number_of_features,adj):
       super(SGCN,self).__init__()
       self.args=args
       self.number_of_labels=number_of_labels
       self.number_of_features = number_of_features
       self.device = args.device
       self.adj=sparse_mx_to_torch_sparse_tensor(adj).to(self.device).to_dense().matrix_power(args.K).float()
       self.setup_layers()
    def setup_layers(self):
       self.layer_2 = DenseFullyConnected(self.args.ne, self.number_of_labels)
    def forward(self,feature_indices, feature_values):
       
       x=torch.sparse.FloatTensor(feature_indices,feature_values,torch.Size([self.args.nu,self.args.ne])).to(self.device).to_dense().float()
       x=self.layer_2(x)
       x=torch.mm(self.adj,x)
       return F.log_softmax(x, dim=1)
