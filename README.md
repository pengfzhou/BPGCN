# BPGCN

benchmarks for gcns

if you want to use pre_processed bpgcn version on pubmed  you can run:

python bpgcn_real.py  --epoch 200    --early_stop 25 --eps1 0.1 --eps2 0.6 --cuda 2 --netdepth 12 --beta 0.5 --dataset pubmed --vanilla 0 --bib 0

Also  you can get  classification of cora and  citeseer using :

python bpgcn_real.py  --epoch 200    --early_stop 25 --eps1 0.1 --eps2 0.6 --cuda 2 --netdepth 5 --beta 2 --dataset cora

python bpgcn_real.py  --epoch 200    --early_stop 25 --eps1 0.1 --eps2 0.5 --cuda 2 --netdepth 5 --beta 2 --dataset citeseer

if you want to get Fig.4 you can run :

chmod 777 run_syn.sh(maybe not needed)

./run_syn.sh 

BP  with C++ can be run in the file JSBM_C++ with command:

./test1.sh

for Fig3, may using MPI for multipul instances on varying parameters.
