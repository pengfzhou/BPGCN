import argparse
from math import sqrt

parser = argparse.ArgumentParser()
group = parser.add_argument_group('physics parameters')
group.add_argument('--eps1',type=float,default=0.3,help='cout/cin')
group.add_argument('--eps2',type=float,default=0.3,help='wout/win')
group.add_argument('--c1',type=float,default=3,help='average degree in unit graph')
group.add_argument('--c2',type=float,default=3,help='average degree in bipartite graph')
group.add_argument('--c3',type=float,default=3,help='average degree in bipartite graph')
group.add_argument('--q',type=float,default=2,help='number of grups')


args = parser.parse_args()
x=args.c2*args.c3*(1-args.eps2)**4/((1+(args.q-1)*args.eps2)**4)
y=args.c1**2*(1-args.eps1)**4/((1+(args.q-1)*args.eps1)**4)
print(2*x+y+sqrt(y*(4*x+y)))
print(x+sqrt(y))


