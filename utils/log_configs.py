import os
import os.path as osp

def log_configs(args, logdir):
    line = ' '.join(f'{k}={v}' for k, v in vars(args).items()) 
    print(line)
    if not osp.isdir(logdir):
        os.makedirs(logdir)
    with open(osp.join(logdir, 'config-log.txt'), 'a') as f:
        f.write(line)