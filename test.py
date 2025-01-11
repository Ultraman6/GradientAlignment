import os
import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'

dist.init_process_group('gloo', rank=0, world_size=1, init_method="env://?use_libuv=False")