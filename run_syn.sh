#!/bin/bash
var1=50
var2=2
for ((i=1;i<=10;i+=1))
 do
 var2=$((var2+100))
 for ((j=1;j<=50;j+=1))
 do
   vars=$(echo "scale=4; 0+$j / $var1" | bc)
   #./sbm infer  -n1000 -b 1000 -q2 -P0.3,3.0  -a $vars,3.0   -t 800   -i2 -R0.85
   python main_gcn.py  --epoch 200 --ne 10000 --nu 10000 --K 5 --c1 10  --c2 10 --eps1 0.1 --eps2 $vars --Q 5 --hidden 64 --wd 5e-4 --rho 0.05 --seed 0 --seed_model $var2 --alpha 0.2 --net sgcn --early_stop 25 --nheads 4 --dropout 0.5 --lr 0.005 --fname sgcn_c1=10_c2=10__eps2.txt --cuda 5

 done
   


done