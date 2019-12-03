#!/bin/bash
var1=2
var2=2
for ((i=1;i<=5;i+=1))
do
 #var2=$((var2*10))
 for ((j=1;j<20;j+=1))
 do
   vars=$(echo "scale=4; 10+$j / $var1" | bc)
   ./sbm infer  -n10000 -b 10000 -q5 -P0.1,$vars  -a 0.2,6  -t 1000   -i1 -R0.95 -M bpc2=6c1.txt  -f 0.05  -e0.00001
   
 done
   


done
