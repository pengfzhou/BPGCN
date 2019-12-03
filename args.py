import argparse

parser = argparse.ArgumentParser()

group = parser.add_argument_group('physics parameters')
group.add_argument(
    '--seed_model',
    type=int,
    default=1,
    help='random seed for generation, 0 for randomized')
group.add_argument(
    '--model',
    type=str,
    default='jsbm',
    choices=['jsbm'],
    help='generative model')
group.add_argument('--nu', type=int, default=10000, help='number of data points')
group.add_argument('--ne', type=int, default=1000, help='number of metadata points')
group.add_argument('--Q', type=int, default=2, help='number of groups')
group.add_argument('--eps1',type=float,default=0.1,help='cout/cin')
group.add_argument('--eps2',type=float,default=0.1,help='wout/win')
group.add_argument('--c1',type=float,default=4,help='average degree in unit graph')
group.add_argument('--c2',type=float,default=4,help='average degree in bipartite graph')
group.add_argument('--dataset',type=str,default='cora',help='which kind of dataset')
group.add_argument('--dump',type=float,default=1.0,help='dumping rate')
group.add_argument('--split',type=int,default=0,help='split setting')
group.add_argument('--rho',type=float,default=0.052,help='label rate')
group.add_argument('--bib',type=int,default=1,help='with or without bibpartite graph ,1 means with')
group = parser.add_argument_group('network parameters')

group.add_argument(
    '--dtype',
    type=str,
    default='float32',
    choices=['float32', 'float64'],
    help='dtype')
group.add_argument('--vanilla',type=int,default=1,help='bpgcn version 0 means pre-processed ')
parser.add_argument('--wd',type=float,default=5e-4,help='L2 loss')
group.add_argument('--netdepth',type=int,default=5,help='numbers of bp_layers')
group.add_argument('--alpha',type=float,default=1.0,help=' initial weight')
group.add_argument('--beta',type=float,default=1.0,help='field strengeth')
group.add_argument('--hidden',type=int,default=32,help='number of hidden layers')
group.add_argument('--dropout',type=float,default=0.5,help='dropout rate')
group.add_argument('--x',type=float,default=0.5,help='bpgcn initial for eps1')
group.add_argument('--y',type=float,default=0.5,help='bpgcn initial for eps2')


group.add_argument(
    '--epsilon',
    type=float,
    default=1e-7,
    help='small number to avoid 0 in division and log')

group = parser.add_argument_group('iterate parameters')
group.add_argument(
    '--seed', type=int, default=0, help='random seed, 0 for randomized')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--net', type=str, default='GCN',
                    help='GCN_like net.')
parser.add_argument('--nheads',type=int,default=8,help="Number of head attentions")
parser.add_argument('--K',type=int,default=6,help='powers of adjacency matrix')
group.add_argument(
    '--optimizer',
    type=str,
    default='adam',
    help='optimizer')
group.add_argument(
    '--init_flag', type=int, default=1, help='initials for x')
group.add_argument(
    '--lr', type=float, default=0.01, help='learning rate')
group.add_argument(
    '--max_iter_time', type=int, default=2*(10**2), help='maximum number of iterative times')
group.add_argument(
    '--early_stop', type=int, default=10, help='early_stopping')

group.add_argument(
    '--conv_crite',
    type=float,
    default=1e-3,
    help='convergeing criteria ')

group = parser.add_argument_group('system parameters')
group.add_argument(
    '--iter_time', type=int, default=10, help='maximum number of iterative times')
group.add_argument(
    '--no_stdout',
    action='store_true',
    help='do not print log to stdout, for better performance')
group.add_argument(
    '--clear_checkpoint', action='store_true', help='clear checkpoint')
group.add_argument(
    '--print_time',
    type=int,
    default=1,
    help='number of times to print log, 0 for disabled')
group.add_argument(
    '--save_time',
    type=int,
    default=100,
    help='number of times to save model, 0 for disabled')

group.add_argument(
    '--print_sample',
    type=int,
    default=1,
    help='number of samples to print to log on visual_time, 0 for disabled')

group.add_argument(
    '--cuda', type=int, default=0, help='ID of GPU to use, -1 for disabled')
group.add_argument(
    '--out_infix',
    type=str,
    default='',
    help='infix in output filename to distinguish repeated runs')
group.add_argument(
    '-o',
    '--out_dir',
    type=str,
    default='out',
    help='directory prefix for output, empty for disabled')
group.add_argument(
    '--fname',
    type=str,
    default='out.txt',
    help='file name to append the results to')

args = parser.parse_args()
