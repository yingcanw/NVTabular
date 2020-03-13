import os
import argparse
import torch
import pandas as pd
import numpy as np
import time
import sys
sys.path.insert(1, '../')

def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("gpu_id", help="gpu index to use")
    parser.add_argument("in_dir", help="directory with dataset files inside")
    parser.add_argument("in_file_type", help="type of file (i.e. parquet, csv, orc)")
    return parser.parse_args()


args = parse_args()
print(args)
GPU_id = args.gpu_id
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)



from fastai import *
from fastai.basic_data import *
from fastai.basic_data import *
from fastai.tabular import *
from fastai.basic_data import DataBunch
from fastai.tabular import TabularModel

import cudf


import nv_tabular as nvt
from nv_tabular.preproc import Workflow
from nv_tabular.ops import Normalize, FillMissing, Categorify, Moments, Median, Encoder, LogOp, ZeroFill
from nv_tabular.dl_encoder import DLLabelEncoder
from nv_tabular.ds_iterator import GPUDatasetIterator
from nv_tabular.batchloader import FileItrDataset, DLCollator, DLDataLoader
import warnings

import matplotlib.pyplot as plt




print(f"torch: {torch.__version__}, cuda: {cudf.__version__}")


data_path = args.in_dir
train_set = [os.path.join(data_path, x) for x in os.listdir(data_path) if x.endswith("parquet")] 
cont_names = ['I' + str(x) for x in range(1,14)]
cat_names =  ['C' + str(x) for x in range(1,24)]
cols = ['label']  + cont_names + cat_names
proc = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=['label'], to_cpu=False)
proc.finalize()
proc.create_final_cols()
dlc = DLCollator(preproc=proc, apply_ops=False)
results = {}
for batch_size in [2**i for i in range(9, 26, 1)]:
    print('Checking batch size: ', batch_size)
    num_iter = max(10 * 1000 * 1000 // batch_size, 100) # load 10e7 samples
    t_batch_sets = [FileItrDataset(x, names=cols, engine=args.in_file_type, batch_size=batch_size, sep="\t") for x in train_set]
    t_chain = torch.utils.data.ChainDataset(t_batch_sets)
    t_data = DLDataLoader(t_chain, collate_fn=dlc.gdf_col, pin_memory=False, num_workers=0)
    
    start = time.time()
    for i, data in enumerate(t_data):
        if i >= num_iter:
            break
    stop = time.time()

    throughput = num_iter * batch_size / (stop - start)
    results[batch_size] = throughput
    print('batch size: ', batch_size, ', throughput: ', throughput)