#!/bin/bash
var1=50
var2=2
for ((i=1;i<=10;i+=1))
do
 #var2=$((var2+10))
 for ((j=1;j<=50;j+=1))
 do
   vars=$(echo "scale=4; 0+$j / $var1" | bc)
   ./sbm infer  -n10000 -b 10000 -q5 -P 0.4,10  -a $vars,10  -t 1000   -i1 -R0.95 -M bpc1=10c2=10_eps1=0.4_eps2.txt  -f 0.05  -e0.00001
   
 done
   


done
