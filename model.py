
from math import exp, log, pi, sqrt,floor

import h5py
import numpy as np
import torch
from scipy.integrate import nquad
from torch.nn import functional as F
import random
from args import args 

from utils1 import assert_not_nan, ensure_dir, random_state

def coord_to_idx(i, j, L):
    return (i % L) * L + (j % L)


def ln2cosh(x):
    return F.softplus(2 * x) - x
def _sigmoid(x,rho):
        return rho / ((1-rho)*torch.exp(-x) + rho)

class sbm:
    def __init__(self, **kwargs):
        self.ham = kwargs['model']
        self.nu = kwargs['nu']
        self.ne = kwargs['ne']
   
        self.seed = kwargs['seed_model']
        self.device = kwargs['device']
       
        self.eps1 = kwargs['eps1']
        self.eps2 = kwargs['eps2']
        
        self.c1 = kwargs['c1']
        self.c2 = kwargs['c2']
        self.Q = kwargs['Q']
        
        self.C=torch.zeros([self.Q,self.Q]).to(self.device)
        self.W=torch.zeros([self.Q,self.Q]).to(self.device)
        self.conf_true=torch.zeros(self.nu+self.ne).long().to(self.device)
        

    def generate_sbm(self):
        random.seed(self.seed)
        #comput C and W matrix
        cin=self.Q*self.c1/(1+(self.Q-1)*self.eps1)
        win=self.Q*self.c2/(1+(self.Q-1)*self.eps2)
        cou=cin*self.eps1
        wou=win*self.eps2
        for i in range(self.Q):
            for  j  in range(self.Q):
               if i==j:
                  self.C[i][j]=cin
                  self.W[i][j]=win
               else:
                  self.C[i][j]=cou
                  self.W[i][j]=wou
        #generate ground truth3
        print(self.C)
        for q in range(self.Q):   #data point groundtruth
            for j in range(int(self.nu/self.Q)):
               self.conf_true[int(q*self.nu/self.Q)+j]=q
        
        for q in range(self.Q):  # metadata point groundtruth
            for j in range(int(self.ne/self.Q)):
               self.conf_true[self.nu+int(q*self.ne/self.Q)+j]=q
        labels=self.conf_true[0:self.nu]
        #generate edge
        E1=[] #store edges from i to j
        graph_neis1=[[] for i in range(self.nu)]
        E2=[] #store edges from i to a
        graph_neis2=[[] for i in range(self.ne+self.nu)]
        E3=[] #store edges from a to i 
        for q1 in range (self.Q):
            for q2 in range (q1,self.Q):
               if q1==q2:
                 numlinks1=int(cin/self.Q*(self.nu/self.Q-1)/2) # numlinks in unigraph
                 print(numlinks1)
               else:
                 numlinks1=int(cou*self.nu/(self.Q**2))
                 print(numlinks1)

               for i  in range(numlinks1):
                   bad=True
                   while(bad):
                         bad =False
                         v1=random.randint(int(q1*self.nu/self.Q), int((q1+1)*self.nu/self.Q)-1)
                         v2=random.randint(int(q2*self.nu/self.Q), int((q2+1)*self.nu/self.Q)-1)
                         if v1==v2:
                            bad=True
                         if (not bad):
                            for p in range(len(graph_neis1[v1])):
                                if (graph_neis1[v1][p]==v2):
                                   bad=True;
                                   break;
                         
                         if (not bad):
                            for p in range(len(graph_neis1[v2])):
                                if (graph_neis1[v2][p]==v1):
                                   bad=True;
                                   break;
                         
                   graph_neis1[v1].append(v2)
                   graph_neis1[v2].append(v1)
                   E1.append([v1,v2])
                   E1.append([v2,v1])
        print(len(E1))
        for q1 in range (self.Q):
            for q2 in range (self.Q):
               if q1==q2:
                 numlinks2=int(win*self.nu/(self.Q**2)) # numlinks in bigraph
               else:
                 numlinks2=int(wou*self.nu/(self.Q**2))
               print(numlinks2)
               for i  in range(numlinks2):
                   bad=True
                   while(bad):
                         bad =False
                         v1=random.randint(q1*self.nu/self.Q, (q1+1)*self.nu/self.Q-1)
                         v2=random.randint(self.nu+q2*self.ne/self.Q, self.nu+(q2+1)*self.ne/self.Q-1)
                         if v1==v2:
                            bad=True
                         if (not bad):
                            for p in range(len(graph_neis2[v1])):
                                if (graph_neis2[v1][p]==v2):
                                   bad=True;
                                   break;
                         '''
                         if (not bad):
                            for p in range(len(graph_neis2[v2])):
                                if (graph_neis2[v2][p]==v1):
                                   bad=True;
                                   break;
                         '''   
                   graph_neis2[v1].append(v2)
                   graph_neis2[v2].append(v1)
                   E2.append([v1,v2])
                   E3.append([v2,v1])
        
        return np.array(E1),np.array(E2),np.array(E3),labels               
                   
    def init(self,nu,ne,E1,E2,E3,Q,onehot_label,idx_train):
        if args.init_flag==1:
           marg_i=1/Q*torch.ones(nu,Q)
           marg_a=1/Q*torch.ones(ne,Q)
           cav_ij=1/Q*torch.ones(E1.shape[0],Q)
           cav_ia=1/Q*torch.ones(E2.shape[0],Q)
           cav_ai=1/Q*torch.ones(E3.shape[0],Q) 
           field_i=marg_i
           field_ij=cav_ij
           field_ia=cav_ia
        elif args.init_flag==2:
           marg_i=F.softmax(torch.rand(nu,Q),1)                      
           marg_a=F.softmax(torch.rand(ne,Q),1)  
           cav_ij=F.softmax(torch.rand(E1.shape[0],Q),1) 
           cav_ia=F.softmax(torch.rand(E2.shape[0],Q),1)    
           cav_ai=F.softmax(torch.rand(E3.shape[0],Q),1) 
           field_i=marg_i
           field_ij=cav_ij
           field_ia=cav_ia

        elif args.init_flag==3:
           onehot_label1=torch.zeros(nu+ne, Q).scatter_(1, self.conf_true.cpu(), 1)
           E1_=torch.from_numpy(E1).long()
           A1=torch.zeros(E1.shape[0], nu).scatter_(1, E1_[:,0:1], 1)
           E2_=torch.from_numpy(E2).long()
           A2=torch.zeros(E2.shape[0], nu).scatter_(1, E2_[:,0:1], 1)
           E3_=torch.from_numpy(E3).long()
           A3=torch.zeros(E3.shape[0], ne).scatter_(1, E3_[:,0:1]-nu, 1)
           marg_i=onehot_label1[0:nu]
           marg_a=onehot_label1[nu:nu+ne]
           cav_ij=A1 @ marg_i
           cav_ia=A2 @ marg_i
           cav_ai=A3 @ marg_a
           field_i=marg_i
           field_ij=cav_ij
           field_ia=cav_ia
        elif args.init_flag==4:
              
           E1_=torch.from_numpy(E1).long()
           A1=torch.zeros(E1.shape[0], nu).scatter_(1, E1_[:,0:1], 1)
           E2_=torch.from_numpy(E2).long()
           A2=torch.zeros(E2.shape[0], nu).scatter_(1, E2_[:,0:1], 1)
           marg_i=F.softmax(torch.ones(nu,Q),1)                      
           marg_a=F.softmax(torch.ones(ne,Q),1) 
           marg_i[idx_train]=onehot_label[idx_train]
           cav_ij=A1 @ marg_i
           cav_ia=A2 @ marg_i
           cav_ai=F.softmax(torch.ones(E3.shape[0],Q),1)
           field_i=1/Q*torch.ones(nu,Q)
           field_i[idx_train]=torch.abs(onehot_label[idx_train]-0.1)
           field_ij=A1 @ field_i
           field_ia=A2 @ field_i
        
        else:
           print('no initial flag')
        return marg_i,marg_a,cav_ij,cav_ia,cav_ai,torch.log(field_i),torch.log(field_ij),torch.log(field_ia)
    def opt_init_pre(self,nu,ne,A1,A2,A3,Q,onehot_label,idx_train,feature):
           
        marg_i=F.softmax(feature,1)   
        field_i=F.softmax(feature,1)              
        marg_a=F.softmax(torch.ones(ne,Q),1) 
        #marg_i[idx_train]=onehot_label[idx_train]
        cav_ij=A1.cpu() @ marg_i
        cav_ia=A2.cpu() @ marg_i
        cav_ai=F.softmax(torch.ones(A3.shape[0],Q),1) 
           
        return marg_i,marg_a,cav_ij,cav_ia,cav_ai,torch.log(field_i)
        
    def opt_init(self,nu,ne,A1,A2,A3,Q,onehot_label,idx_train):
        marg_i=F.softmax(torch.ones(nu,Q),1) 
        field_i=1/Q*torch.ones(nu,Q)
        field_i[idx_train]=torch.abs(onehot_label[idx_train]-0.1)
        marg_a=F.softmax(torch.ones(ne,Q),1) 
        marg_i[idx_train]=onehot_label[idx_train]
        cav_ij=A1.cpu() @ marg_i
        cav_ia=A2.cpu() @ marg_i
        cav_ai=F.softmax(torch.ones(A3.shape[0],Q),1)
        return marg_i,marg_a,cav_ij,cav_ia,cav_ai,torch.log(field_i)
        
    def init_unit_bp(self,nu,E1,Q,onehot_label,idx_train):
        E1_=torch.from_numpy(E1).long()
        A1=torch.zeros(E1.shape[0], nu).scatter_(1, E1_[:,0:1], 1)
           
        marg_i=F.softmax(torch.ones(nu,Q),1)                      
        #marg_a=F.softmax(torch.ones(ne,Q),1) 
        marg_i[idx_train]=onehot_label[idx_train]
        cav_ij=A1 @ marg_i
        field_i=1/Q*torch.ones(nu,Q)
        field_i[idx_train]=torch.abs(onehot_label[idx_train]-0.1)
        
        return marg_i,cav_ij,torch.log(field_i)
    def init_nmf(self,nu,ne,Q,onehot_label,idx_train):
           
        marg_i=F.softmax(torch.ones(nu,Q),1)
        marg_a=F.softmax(torch.ones(ne,Q),1)
        marg_i[idx_train]=onehot_label[idx_train]

        field_i=1/Q*torch.ones(nu,Q)
        field_i[idx_train]=torch.abs(onehot_label[idx_train]-0.1)
        return marg_i,marg_a,torch.log(field_i)
    
    

    
    

    
