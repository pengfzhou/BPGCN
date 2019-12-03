#include<iostream>

#include<Python.h>
using namespace std;
extern "C"{
PyObject  * test(int x,int y,int *matrix, int rows, int columns,int*matrix1,int rows1)
{       
         //  g++ -o test2.so -shared -fPIC test2.cpp -I/home/zpf/anaconda3/include/python3.6m
        PyObject* result = PyList_New(0);
       // PyList_Append(result, PyLong_FromLong(1));
        
        

	//printf("Hello World!\n");
        if(x==0){
                for (int i=0;i<rows;i++){
                    for (int j=0;j<rows1;j++){
                       
                       if (matrix[i*columns+0]==matrix1[j*columns+1] && matrix[i*columns+1]!=matrix1[j*columns+0])
                        
                        {
                         PyList_Append(result, PyLong_FromLong(i));
                         PyList_Append(result, PyLong_FromLong(j));
                         num++;
                         printf(num);
                        }
                     }
                }
              }
        else{ 
               
                for (int i=0;i<x;i++){
                    for (int j=0;j<rows;j++){
                       if (i==matrix[j*columns+1])
                        {PyList_Append(result, PyLong_FromLong(i));
                         PyList_Append(result, PyLong_FromLong(j));
                        }
                   }
                }
	
	    }
        
        
	return result;
}
}
