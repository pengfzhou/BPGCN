import os    


for i in range(5):
    os.system("./sbm infer  -n1000000 -b 1000000 -q2 -P0.3,3.0  -a0.48,3.0   -t 1000   -i2   -R0.85")
for i in range(5):
    os.system("./sbm infer  -n1000000 -b 1000000 -q2 -P0.3,3.0  -a0.48,3.0   -t 1000   -i1  -R0.85 " )
for i in range(5):
    os.system("./sbm infer  -n1000000 -b 1000000 -q2 -P0.3,3.0  -a0.50,3.0   -t 1000   -i1  -R0.85 " )
for i in range(5):
    os.system("./sbm infer  -n1000000 -b 1000000 -q2 -P0.3,3.0  -a0.50,3.0   -t 1000   -i2  -R0.85 " )
