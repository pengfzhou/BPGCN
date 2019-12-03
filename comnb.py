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
def ShowCArrayArray(caa):
    for row in caa:
        for col in row:
            print (col)
def convert(x):
    y=np.array(x)
    z=torch.from_numpy(y).long().view(-1,2)
    return z
def ret_opera(E1,nu):
    M1=E1.shape[0]
    
   
    c_E1_arr = Convert2DToCArray(E1)
    ShowCArrayArray(c_E1_arr)
    so = CDLL("./test2.so")
 
    
   
    so.test.restype = py_object
    #B=so.test(0,0,c_E1_arr,M1,2,c_E1_arr,M1)
    #print(len(B))
    B_=so.test(nu,0,c_E1_arr,M1,2)
    
    #print(len(B_))
    
    return convert(B_)
    

