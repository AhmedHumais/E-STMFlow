from __future__ import print_function, division

import random

import torch
import torch.nn as nn

from tqdm import tqdm

from utils import flow_viz
import numpy as np
import imageio
import os

import sys
from pathlib import Path

import argparse
import numpy as np
import json
from torch.utils.data.dataloader import DataLoader
from model.emamba import EMambFlow

from loader.loader_mamba_dsec_test import Sequence
from utils.dsec_utils import RepresentationType, VoxelGrid
from collections import OrderedDict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model(ckpt):
    mdl = torch.load(ckpt, weights_only=True)
    mdl2 = OrderedDict()
    for key, v in mdl.items():
        mdl2[key.replace("model.","")] = v
        
    return mdl2

def load_args(filename: str):
    args = argparse.Namespace()
    with open(filename, 'r') as f:
        args.__dict__ = json.load(f)

def test(checkpt, loader, result_dir):

    device = 0
    model = EMambFlow(load_args(f'configs/m_x2.json'))    
    print("Parameter Count: %d" % count_parameters(model))

    model.load_state_dict(load_model(checkpt), strict=True)

    model.to(device=torch.device(device))
    model.eval()
    
    idx = 0
    for batch in tqdm(loader):
        ev = batch['event_volume']

        ev = ev.to(device = device, dtype = torch.float32)

        with torch.no_grad():
            flow, _ = model(ev)            


        flow = flow[0].cpu().numpy()
        
        flo = flow_viz.flow_to_image(flow.transpose(1,2,0))
        
        flow_path = os.path.join(result_dir, 'vis', str(idx).zfill(6)+'.png')        
        imageio.imwrite(flow_path, flo, format = 'PNG-FI')

        if batch['save_submission'] == True:
            sub_path = os.path.join(result_dir, str(batch['file_index'].item()).zfill(6)+'.png')
            _, h,w = flow.shape
            flow_map = np.rint(flow*128 + 2**15)
            flow_map = flow_map.astype(np.uint16).transpose(1,2,0)
            I = np.concatenate((flow_map, np.zeros((h,w,1), dtype=np.uint16)), axis=-1)
            imageio.imwrite(sub_path, I, format = 'PNG-FI')
                               
        idx += 1

    return


if __name__ == '__main__':
    model_name = 'm1x2_v1'
    ckpt = f'trained_models/{model_name}.pth'
    data_path = 'DSEC_Dataset/test/'
    results_path = f'ESTMFlowResults/{model_name}'    

    seqs = sorted(os.listdir(data_path))
    
    seq = seqs[0]
    for seq in seqs:
        result_path = os.path.join(results_path, seq)
        if not os.path.isdir(os.path.join(result_path, 'vis')):
            os.makedirs(os.path.join(result_path, 'vis'))
        test_set = Sequence(Path(data_path+ seq), 0)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=8, shuffle = False, drop_last = True, pin_memory = True)
        test(ckpt, test_loader, result_path)        
