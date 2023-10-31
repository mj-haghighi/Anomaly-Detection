import os
import sys
import wget
import argparse
import os.path as osp
sys.path.append('.')

from configs import configs
from enums.ext import EXT
from utils.extract import extract

def download_dataset(dataset_name: str, outdir=None):
    if dataset_name not in configs.keys():
        raise Exception("Unknown dataset '{}'".format(dataset_name))
    
    config = configs[dataset_name]
    outdir = osp.join(config.outdir, dataset_name) if outdir is None else outdir
    if osp.isdir(outdir):
        print("Dataset already exist in {}".format(osp.join(outdir)))
        return
    
    if not osp.isdir(outdir):
        os.makedirs(outdir)

    outpath = osp.join(outdir, dataset_name+"."+config.filetype)
    wget.download(url=config.download_link, out=outpath)
    extract(path=outpath, to=outdir, type=config.filetype)

##: To use directly
def parse_args():
    parser = argparse.ArgumentParser(description='download dataset')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cfar10', 'cfar100'], help='choose dataset')

    args = parser.parse_args()
    return args

def main(argv=None):
    args = parse_args()
    download_dataset(dataset_name=args.dataset)

if __name__ == "__main__":    
    main()