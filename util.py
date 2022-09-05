import os
import datetime
from pathlib import Path
import random
import numpy as np
import torch
import torch.distributed as dist

save_id = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
log_dir = Path(f"runs") /  f"anomaly-{save_id}" /"logs"
if int(os.environ['WORLD_SIZE']) > 1:
    os.makedirs(log_dir,exist_ok=True)
    f = open(log_dir/'log.txt','a')

def pprint(*contant):
    if int(os.environ['WORLD_SIZE']) > 1:
        if int(os.environ['LOCAL_RANK']) == 0:
            print(*contant)
            f.write(f'{"".join(str(contant))}\n')
            f.flush()
    else:
        print(*contant)

def add_salt_pepper(x:torch.FloatTensor,min_value,max_value,sigma,p=0.):
    assert isinstance(x,torch.FloatTensor)
    nb_salt = random.randint(0,int(x.numel()*sigma))
    nb_pepper = random.randint(0,int(x.numel()*sigma))
    x_shape = x.shape
    if random.random() < p:
        salt_indices, pepper_indices = torch.split(torch.randperm(x.numel())[:(nb_salt+nb_pepper)],[nb_salt,nb_pepper])
        x = x.flatten()
        x[salt_indices] = min_value
        x[pepper_indices] = max_value
        x = torch.reshape(x,x_shape)
        y = x.clone()
    else:
        y = x.clone()
    return y

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        pprint('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    if 'win' in os.environ['OS'].lower():
        args.dist_backend = 'gloo'  # 通信后端，nvidia GPU推荐使用NCCL，如果报错使用gloo
    elif 'ubuntu' in os.environ['OS'].lower():
        args.dist_backend = 'nccl'
    else:
        raise EnvironmentError
    pprint('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url))
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()

def setup_DDP(backend="nccl", verbose=False):
    """
    We don't set ADDR and PORT in here, like:
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
    Because program's ADDR and PORT can be given automatically at startup.
    E.g. You can set ADDR and PORT by using:
        python -m torch.distributed.launch --master_addr="192.168.1.201" --master_port=23456 ...

    You don't set rank and world_size in dist.init_process_group() explicitly.

    :param backend:
    :param verbose:
    :return:
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # If the OS is Windows or macOS, use gloo instead of nccl
    pprint('rank: ',rank)
    pprint('local rank: ',local_rank)
    pprint('world size: ',world_size)
    dist.init_process_group(backend=backend)
    # set distributed device
    device = torch.device("cuda:{}".format(local_rank))
    if verbose:
        pprint("Using device: {}".format(device))
        pprint(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
    return rank, local_rank, world_size, device

def indices2logic(m:np.array,size:int):
    n = np.zeros(size,dtype='bool')
    n[m] = True
    return n