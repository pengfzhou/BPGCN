from ctypes import *
import numpy as np
import ctypes
import torch
def Convert1DToCArray(TYPE, ary):
    arow = TYPE(*ary.tolist())
    return arow
 
def Convert2DToCArray(ary):
    ROW = c_int * len(ary[0])
    rows = []
    for i in range(len(ary)):
        rows.append(Convert1DToCArray(ROW, ary[i]))
    MATRIX = ROW * len(ary)
    return MATRIX(*rows)
def convert(x):
    y=np.array(x)
    z=torch.from_numpy(y).long().view(-1,2)
    return z
def return_opera(E1,E2,E3,nu,ne):
    M1=E1.shape[0]
    M2=E2.shape[0]
    M3=E3.shape[0]
    E2[:,1:2]=E2[:,1:2]+nu
    E3[:,0:1]=E3[:,0:1]+nu
    print(M1)
    print(M2)
    print(M3)
    c_E1_arr = Convert2DToCArray(E1)
    c_E2_arr = Convert2DToCArray(E2)
    c_E3_arr = Convert2DToCArray(E3)

    so = CDLL("./test2.so")
 
    #INPUT = c_int * 2
    #INPUT2=INPUT * 2
    #input = INPUT()
    #input[0] = 1
    #input[1] = 2
   
    so.test.restype = py_object
    B=so.test(0,0,c_E1_arr,M1,2,c_E1_arr,M1)
    print(len(B))
    # print(convert(B))
    B_=so.test(nu,0,c_E1_arr,M1,2)
    print(len(B_))
    # print(convert(B_))
    R_=so.test(nu,0,c_E3_arr,M3,2)
    #print(len(R_))
    Q_=so.test(ne,nu,c_E2_arr,M2,2)
    print(len(Q_))
    #print(convert(Q_))
    N= so.test(0,0,c_E1_arr,M1,2,c_E3_arr,M3)
    print(len(N))
    
    Q=so.test(0,0,c_E3_arr,M3,2,c_E2_arr,M2)
    print(len(Q))
    M=so.test(0,0,c_E2_arr,M2,2,c_E1_arr,M1)
    print(len(M))
    R=so.test(0,0,c_E2_arr,M2,2,c_E3_arr,M3)
    print(len(R))
    
    return convert(B),convert(B_),convert(M),convert(N),convert(R),convert(R_),convert(Q),convert(Q_)
    
    #print(p)
    #print(np.array(p))
def return_opera_1(E1,E2,E3,nu,ne):
    M1=E1.shape[0]
    M2=E2.shape[0]
    M3=E3.shape[0]
    E2[:,1:2]=E2[:,1:2]+nu
    E3[:,0:1]=E3[:,0:1]+nu
    print(M1)
    print(M2)
    print(M3)
    c_E1_arr = Convert2DToCArray(E1)
    c_E2_arr = Convert2DToCArray(E2)
    c_E3_arr = Convert2DToCArray(E3)
    so = CDLL("./test2.so")
 
    so.test.restype = py_object
    
    B_=so.test(nu,0,c_E1_arr,M1,2)
    print(len(B_))
    # print(convert(B_))
    R_=so.test(nu,0,c_E3_arr,M3,2)
    #print(len(R_))
    Q_=so.test(ne,nu,c_E2_arr,M2,2)
    print(len(Q_))
    return convert(B_),convert(R_),convert(Q_)
    
  

