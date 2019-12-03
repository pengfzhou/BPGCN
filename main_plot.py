import torch
import time
import random
import itertools
import numpy as np
from args import args
from model import sbm 
import torch.nn.functional as F
from compNB import return_opera
from comnb  import ret_opera

from utils1 import (
    ensure_dir,
    init_out_dir,
    my_log,
    print_args,
    
    
)
from utils import load_realdata
from unit_bpnet import Bpnet
def accuracy(output, labels):
    codes=output.max(1)[1].type_as(labels).to(args.device)
    clusters=labels.max().item()+1
    s=[]
    for i in range(clusters):
        s.append(i)
    perms=list(itertools.permutations(s,clusters))
    perms=torch.from_numpy(np.array(perms)).to(args.device)
    preds=perms[:,codes]
    correct = codes.eq(labels).double()
    correct = torch.sum(correct)
    
    return correct / len(labels) 
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
    print_args()
    E1,labels,onehot_label,idx_train, idx_val, idx_test=load_realdata(args.nu)
    args.Q=labels.max().item() + 1
    
    model= sbm(**vars(args))
    m1=torch.from_numpy(E1).long()
    A=torch.sparse.FloatTensor(m1.t(), torch.ones(m1.shape[0]), torch.Size([args.nu,args.nu])).to(args.device).to_dense()
    args.c1=torch.mean(torch.sum(A,1))
    E1=np.argwhere(np.array(A.cpu())==1)
    indice1=torch.ones(E1.shape[0],2).long()
    
    indice1[:,0:1]=torch.LongTensor(np.arange(E1.shape[0]).reshape(E1.shape[0],1))
    
    indice1[:,1:2]=torch.LongTensor(E1[:,0:1])
    
    A1=torch.sparse.FloatTensor(indice1.t(), torch.ones(indice1.shape[0]), torch.Size([indice1.shape[0],args.nu])).to(args.device)
    
    read_start_time=time.time()
    if(args.dataset != "pubmed"):    
       B_=ret_opera(E1,args.nu)
    else:
        # output_ope(E1,E2,E3,nu,ne)
       #B=torch.from_numpy(readmatrix_g("pub_B.txt")).long()
       B_=torch.from_numpy(readmatrix_g("pub_B_.txt")).long()
       #M=torch.from_numpy(readmatrix_g("pub_M.txt")).long()
       #N=torch.from_numpy(readmatrix_g("pub_N.txt")).long()
       #R=torch.from_numpy(readmatrix_g("pub_R.txt")).long()
       #R_=torch.from_numpy(readmatrix_g("pub_R_.txt")).long()
       #Q=torch.from_numpy(readmatrix("pub_Q.txt")).long()
       #Q_=torch.from_numpy(readmatrix_g("pub_Q_.txt")).long()
    read_end_time=time.time()
    print("Total time for reading: {:.4f}s".format(read_end_time - read_start_time))
    '''
    np.savetxt("B.txt",B.numpy())
    np.savetxt("B_.txt",B_.numpy())
    np.savetxt("M.txt",M.numpy())
    np.savetxt("N.txt",N.numpy())
    np.savetxt("R.txt",R.numpy())
    np.savetxt("R_.txt",R_.numpy())
    np.savetxt("Q_.txt",Q_.numpy())
    np.savetxt("Q.txt",Q.numpy())
    
    B=torch.from_numpy(np.loadtxt("B.txt")).long()
    B_=torch.from_numpy(np.loadtxt("B_.txt")).long()
    M=torch.from_numpy(np.loadtxt("M.txt")).long()
    N=torch.from_numpy(np.loadtxt("N.txt")).long()
    R=torch.from_numpy(np.loadtxt("R.txt")).long()
    R_=torch.from_numpy(np.loadtxt("R_.txt")).long()
    Q=torch.from_numpy(np.loadtxt("Q.txt")).long()
    Q_=torch.from_numpy(np.loadtxt("Q_.txt")).long()
    print(2)
    '''
    indice_ij=B_[:,1:2].view(-1)
    spB_=con_to_spars1(B_,args.nu,E1)
   
    print(len(E1))
    
    print(args.c1)
    print(len(idx_train))
    print(len(idx_val))
    print(len(idx_test))
    
    marg_i,cav_ij,field_i=model.init_unit_bp(args.nu,E1,args.Q,onehot_label,idx_train)
    labels=labels.to(args.device)
    dx_train=idx_train.to(args.device)
    idx_val=idx_val.to(args.device)
    idx_test=idx_test.to(args.device)
    marg_i=marg_i.to(args.device)
    
    cav_ij=cav_ij.to(args.device)
    
    field_i=field_i.to(args.device)
    
   
    C,W=initCW(args.eps1,args.eps2,args.Q,args.c1,args.c2)
    net = Bpnet(spB_,A1,indice_ij,args.beta*field_i,args.alpha*C)
    net.to(args.device) 
    params = list(net.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = int(sum([np.prod(p.shape) for p in params]))
    print(nparams)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr)
    elif args.optimizer == 'sgdm':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=args.lr, alpha=0.99)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
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
     accuracy=train(epoch,net,optimizer,marg_i,cav_ij,labels,idx_train,idx_val)
     if accuracy >= best_accuracy:
        best_accuracy=accuracy
        test_accu=test(net,marg_i,cav_ij,labels,idx_test)
        step_counter=0
     else:
        step_counter=step_counter+1
        if step_counter>args.early_stop:
           break
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test(net,marg_i,cav_ij,labels,idx_test)
    with open(args.fname, 'a', newline='\n') as f:
            f.write('{}  {:.3g} {:.3g} {}  {:.3g} {:.3g}  {:.3g} {:.3g} \n'.format(
                args.eps1,
                args.eps2,
                args.beta,
                args.netdepth,
                args.c1,
                args.c2,
                test_accu,
                best_accuracy
            ))

def test(net,marg_i,cav_ij,labels,idx_test):
    net.eval()
    output = net(marg_i,cav_ij)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test
def train(epoch,net,optimizer,marg_i,cav_ij,labels,idx_train,idx_val):
    t = time.time()
    net.train()
    optimizer.zero_grad()
    output = net(marg_i,cav_ij)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        net.eval()
        output = net(marg_i,cav_ij)

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
