import torch
import time
import random
import numpy as np
from args import args
from model import sbm
import torch.nn.functional as F
import scipy.sparse as sp
from appnp import GCN,MLP
from appnp import APPNPModel,SpGAT,SGCN
from compNB import return_opera
from utils1 import (
    ensure_dir,
    init_out_dir,
    my_log,
    print_args,
    
    
)
from utils import load_data
#from bpnet import Bpnet

def initCW(eps1,eps2,Q,c1,c2):
    C=torch.zeros([Q,Q]).cpu()
    W=torch.zeros([Q,Q]).cpu()
    cin=Q*c1/(1+(Q-1)*eps1)
    win=Q*c2/(1+(Q-1)*eps2)
    cou=cin*eps1
    wou=win*eps2
    for i in range(Q):
        for  j  in range(Q):
             if i==j:
                C[i][j]=cin
                W[i][j]=win
             else:
                C[i][j]=cou
                W[i][j]=wou
    return C,W
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
'''
python main.py --nu 1000 --ne 1000 --c2 3 --seed_model  50000  --init_flag 2 --max_iter_time 500 --c1 3 --eps1 0.3 --eps2 0.48 --seed 26 --conv_crite 0.0001  --out_infix 00
'''

def con_to_spars(B,E1,E2):
    return torch.sparse.FloatTensor(B.t(), torch.ones(B.shape[0]), torch.Size([E1.shape[0],E2.shape[0]])).to(args.device)
def con_to_spars1(B,n,E2):
    return torch.sparse.FloatTensor(B.t(), torch.ones(B.shape[0]), torch.Size([n,E2.shape[0]])).to(args.device)
       
def main():
    start_time= time.time()
    init_out_dir()
    
    E1, E2,E3, shape2,labels,onehot_label,idx_train, idx_val, idx_test=load_data(args.dataset)
    args.Q=labels.max().item() + 1
    args.nu=shape2[0]
    args.ne=shape2[1]
    print_args()
    #model= sbm(**vars(args))
    
    np.savetxt("labels.txt",labels.numpy(),fmt="%1.0f") 
   # E1,E2,E3,labels=model.generate_sbm()
    labels=labels.to(args.device)
    adj=sp.coo_matrix((np.ones(E1.shape[0]),(E1[:,0],E1[:,1])),shape=(args.nu,args.nu))
   # print(adj)
    
    
    feature_indices=torch.from_numpy(E2).long().t()
    feature_values=torch.ones(E2.shape[0])
    #feature=torch.sparse.FloatTensor(torch.from_numpy(E2).long().t(),torch.ones(E2.shape[0]),torch.Size([args.nu,args.ne])).to_dense().to(args.device).float()
    '''
    randomindex=[i for i in range(args.nu)]
    num_train=int(args.nu*args.rho)
    random.shuffle(randomindex)
    idx_train=randomindex[0:num_train]
    idx_val=randomindex[num_train:num_train+500]
    idx_test=randomindex[num_train+500:args.nu]
    idx_train=torch.LongTensor(idx_train)
    idx_val=torch.LongTensor(idx_val)
    idx_test=torch.LongTensor(idx_test)
    '''
    idx_train=idx_train.to(args.device)
    idx_val=idx_val.to(args.device)
    idx_test=idx_test.to(args.device)
    if args.net == 'GCN':
       net = GCN(args,args.Q,args.ne,adj)
    elif args.net == 'appnp':
       net=APPNPModel(args,args.Q,args.ne,adj)
    elif args.net =='gat':
       net=SpGAT(args,args.Q,args.ne,adj)
    elif args.net=="mlp":
       net=MLP(args,args.Q,args.ne,adj)
    else:
       net=SGCN(args,args.Q,args.ne,adj)
    net.to(args.device) 
    params = list(net.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr)
    elif args.optimizer == 'sgdm':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=args.lr, alpha=0.99)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'adam0.5':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.5, 0.999))
    else:
        raise ValueError('Unknown optimizer: {}'.format(args.optimizer))

    # Train model
    t_total = time.time()
    best_accuracy=0
    step_counter=0
    test_accu=0
    for epoch in range(args.epochs):
        accuracy=train(epoch,net,optimizer,feature_indices, feature_values,labels,idx_train,idx_val)
        if accuracy >=best_accuracy:
           best_accuracy=accuracy
           test_accu=test(net,feature_indices, feature_values,labels,idx_test)
           step_counter=0
        else:
           step_counter=step_counter+1
           if step_counter>args.early_stop:
              break           
        
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test(net,feature_indices, feature_values,labels,idx_test)
    with open(args.fname, 'a', newline='\n') as f:
            f.write('{}  {:.3g}  {:.3g} {:.3g}  {:.3g}\n'.format(
                args.eps1,
                args.eps2,
                args.c1,
                args.c2,
                test_accu
            ))      

def test(net,feature_indices, feature_values,labels,idx_test):
    net.eval()
    output = net(feature_indices, feature_values)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test
def train(epoch,net,optimizer,feature_indices, feature_values,labels,idx_train,idx_val):
    t = time.time()
    net.train()
    optimizer.zero_grad()
    output = net(feature_indices, feature_values)
    np.savetxt('feature_pub.txt',output.detach().cpu().numpy())
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        net.eval()
        output = net(feature_indices, feature_values)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return acc_val.item()




if __name__ == '__main__':
    main()
    
    

