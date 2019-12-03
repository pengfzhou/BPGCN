import glob
import os

import traceback
from contextlib import contextmanager
from glob import glob

import numpy as np
import torch

from args import args

if args.dtype == 'float32':
    default_dtype = np.float32
    default_dtype_torch = torch.float32
elif args.dtype == 'float64':
    default_dtype = np.float64
    default_dtype_torch = torch.float64
else:
    raise ValueError('Unknown dtype: {}'.format(args.dtype))

np.seterr(all='raise')
np.seterr(under='warn')
np.set_printoptions(precision=8, linewidth=160)

torch.set_default_dtype(default_dtype_torch)
torch.set_printoptions(precision=8, linewidth=160)
torch.backends.cudnn.benchmark = True

if not args.seed:
    args.seed = np.random.randint(1, 10**8)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


if args.cuda>=0:
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.cuda)
args.device=torch.device('cpu' if args.cuda<0 else 'cuda:0')


    
def get_model_args_features():
    model_args = '{model}_nu{nu}_c1{c1:g}'
    model_args = model_args.format(**vars(args))

    features = 'ne{ne}_c2{c2:g}'


    features = features.format(**vars(args))

    return model_args, features


def init_out_filename():
    if not args.out_dir:
        return
    model_args, features = get_model_args_features()
    template = '{args.out_dir}/{model_args}/{features}/out{args.out_infix}'
    args.out_filename = template.format(**{**globals(), **locals()})


def ensure_dir(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        try:
            os.makedirs(dirname)
        except OSError:
            pass


def init_out_dir():
    if not args.out_dir:
        return
    init_out_filename()
    ensure_dir(args.out_filename)
    
   


def clear_log():
    if args.out_filename:
        open(args.out_filename + '.log', 'w').close()


def clear_err():
    if args.out_filename:
        open(args.out_filename + '.err', 'w').close()


def my_log(s):
    if args.out_filename:
        with open(args.out_filename + '.log', 'a', newline='\n') as f:
            f.write(s + u'\n')
        if not args.no_stdout:
            print(s)
    else:
        print(s)


def my_err(s):
    if args.out_filename:
        with open(args.out_filename + '.err', 'a', newline='\n') as f:
            f.write(s + u'\n')
        if not args.no_stdout:
            print(s)
    else:
        print(s)


def print_args():
    for k, v in args._get_kwargs():
        my_log('{} = {}'.format(k, v))
    my_log('')



def assert_not_nan(x, *args_):
    if args.cuda >= 0:
        return
    for test_name, test_fn in [
        ('nan', lambda x: torch.isnan(x).any()),
        ('inf', lambda x: (x == float('inf')).any()),
        ('-inf', lambda x: (x == float('-inf')).any()),
    ]:
        try:
            assert not test_fn(x)
        except AssertionError as e:
            my_log('Error: {}, {}'.format(test_name, args_))
            my_log(''.join(traceback.format_stack()))
            raise e

@contextmanager
def random_state(seed):
    last_np_random_state = np.random.get_state()
    last_torch_random_state = torch.get_rng_state()
    if args.cuda >= 0:
        last_torch_cuda_random_state = torch.cuda.get_rng_state_all()

    np.random.seed(seed)
    torch.manual_seed(seed)
    yield

    np.random.set_state(last_np_random_state)
    torch.set_rng_state(last_torch_random_state)
    if args.cuda >= 0:
        torch.cuda.set_rng_state_all(last_torch_cuda_random_state)
