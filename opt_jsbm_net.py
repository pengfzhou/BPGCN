import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
from args import args
from torch.nn import functional as F



class Bplayer(nn.Module):
   
    def __init__(self, C,W,logna,B_,R_,Q_,A1,A2,A3,indice_ai,indice_ia,indice_ij,dump,field_i):
        super(Bplayer, self).__init__()
        
        self.C = Parameter(C)
        self.W=Parameter(W)
        self.bias=Parameter(logna)
        
        self.B_=B_
        self.Q_=Q_
        self.R_=R_
        self.A1=A1
        self.A2=A2
        self.A3=A3
        self.indice_ai=indice_ai
        self.indice_ia=indice_ia
        self.indice_ij=indice_ij
        self.dump=dump
        self.field_i=field_i
        #self.field_ij=field_ij
        #self.field_ia=field_ia
        self.eps=args.epsilon

    
    def forward(self,marg_i,marg_a,cav_ij,cav_ia,cav_ai):
        if (args.bib==1):
           h_i=torch.mean(F.softmax(marg_i,dim=1) @ self.C,0)+torch.mean(F.softmax(marg_a,dim=1)@ self.W)
           h_a=torch.mean(F.softmax(marg_i,dim=1) @ self.W,0)
           temp1=torch.log(cav_ia @ self.W+self.eps)
           marg_a=torch.spmm(self.Q_ , temp1)-h_a
           temp2=torch.log(cav_ij @ self.C+self.eps)
           temp3=torch.log(cav_ai @ self.W +self.eps)
           marg_i=torch.spmm(self.B_ , temp2)+torch.spmm(self.R_,temp3)+self.field_i-h_i
           
           cav_ia=F.softmax(torch.spmm(self.A2 , marg_i) -temp3[self.indice_ia],dim=1)
           #print(cav_ia[0:100])
           cav_ij=F.softmax(torch.spmm(self.A1 , marg_i) -temp2[self.indice_ij],dim=1)
           #print(cav_ij[0:100])
           cav_ai=F.softmax(torch.spmm(self.A3 , marg_a) -temp1[self.indice_ai],dim=1)
           #print(cav_ai[0:10])
        else:
           temp2=torch.log(cav_ij @ self.C+self.eps)
           
           marg_i=torch.spmm(self.B_ , temp2)+self.field_i
           cav_ij=F.softmax(torch.spmm(self.A1 , marg_i) -temp2[self.indice_ij],dim=1)
            
        return marg_i,marg_a,cav_ij,cav_ia,cav_ai
class Bpnet(nn.Module):
    def __init__(self,B_,R_,Q_,A1,A2,A3,indice_ai,indice_ia,indice_ij,field_i,C,W):
        super(Bpnet, self).__init__()
        
        
        self.depth=args.netdepth         
        
        self.dump=args.dump
        self.Q=args.Q
        self.nu=args.nu
        self.device=args.device
        self.logna=torch.log(1.0/self.Q*torch.ones(self.Q)).to(self.device)
       
        self.net2=Bplayer(C,W,self.logna,B_,R_,Q_,A1,A2,A3,indice_ai,indice_ia,indice_ij,self.dump,field_i)
    def forward(self,marg_i,marg_a,cav_ij,cav_ia,cav_ai):
       
        for i in range(self.depth):
               marg_i,marg_a,cav_ij,cav_ia,cav_ai=self.net2(marg_i,marg_a,cav_ij,cav_ia,cav_ai)
        #print(F.softmax(marg_i,dim=1)[0:60])
        #print(F.softmax(marg_i,dim=1)[-60:-1])
        #print(F.softmax(marg_a,dim=1)[0:20])
        #print(F.softmax(marg_a,dim=1)[-20:-1])
        #print("marg_a")
        #print(marg_a[0:500,:])
        return F.log_softmax(marg_i, dim=1)
        












   
