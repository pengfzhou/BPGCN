#!/bin/bash
var1=2
var2=2
for ((i=1;i<=5;i+=1))
do
 #var2=$((var2*10))
 for ((j=1;j<16;j+=1))
 do
   vars=$(echo "scale=4; 2.5+$j / $var1" | bc)
   ./sbm infer  -n10000 -b 10000 -q5 -P0.1,4.0  -a 0.2,$vars   -t 1000   -i1 -R0.95 -M out3.txt  -f 0.01  -e0.00001
   
 done
   


done
