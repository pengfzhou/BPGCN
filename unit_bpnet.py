import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
from args import args
from torch.nn import functional as F



class Bplayer(nn.Module):
   
    def __init__(self, C,logna,B_,A1,indice_ij,dump,field_i):
        super(Bplayer, self).__init__()
        
        self.C = Parameter(C)
        
        self.bias=Parameter(logna)
        
        self.B_=B_
        
        self.A1=A1
        self.indice_ij=indice_ij
        self.dump=dump
        self.field_i=field_i
        #self.field_ij=field_ij
        #self.field_ia=field_ia
        self.eps=args.epsilon

    
    def forward(self,marg_i,cav_ij):
       
        #h_i=torch.mean(F.softmax(marg_i,dim=1) @ self.C,0)
        temp2=torch.log(cav_ij @ self.C+self.eps)
        #print(temp2[0:20])
        #temp3=torch.log(cav_ai @ self.W +self.eps)
        #print(temp3)
        marg_i=torch.spmm(self.B_ , temp2)+self.field_i
        cav_ij=F.softmax(torch.spmm(self.A1 , marg_i) -temp2[self.indice_ij],dim=1)
            
        return marg_i,cav_ij
class Bpnet(nn.Module):
    def __init__(self,B_,A1,indice_ij,field_i,C):
        super(Bpnet, self).__init__()
        
        
        self.depth=args.netdepth         
        
        self.dump=args.dump
        self.Q=args.Q
        self.nu=args.nu
        self.device=args.device
        self.logna=10000*torch.log(1.0/self.Q*torch.ones(self.Q)).to(self.device)
        layers = []
        self.net2=Bplayer(C,self.logna,B_,A1,indice_ij,self.dump,field_i)
    def forward(self,marg_i,cav_ij):
        
        for i in range(self.depth):
               marg_i,cav_ij=self.net2(marg_i,cav_ij)
        #print(F.softmax(marg_i,dim=1)[0:60])
        #print(F.softmax(marg_i,dim=1)[-60:-1])
        #print(F.softmax(marg_a,dim=1)[0:20])
        #print(F.softmax(marg_a,dim=1)[-20:-1])
        #print("marg_a")
        #print(marg_a[0:500,:])
        return F.log_softmax(marg_i, dim=1)
        












   
