from pathlib import Path
import random
from typing import Dict, Tuple
import weakref

import cv2
import h5py
import hdf5plugin
import torch
from torch.utils.data import Dataset
import numpy as np
import imageio.v2 as imageio
import os

from utils.dsec_utils import RepresentationType, VoxelGrid, flow_16bit_to_float
from .loader_mamba import EventSlicer

class Sequence(Dataset):
    def __init__(self, seq_path: Path, seq_idx: int):
        
        assert seq_path.is_dir()
        '''
        Directory Structure:

        Dataset
        └── test
            ├── interlaken_00_b
            │   ├── events_left
            │   │   ├── events.h5
            │   │   └── rectify_map.h5
            │   ├── image_timestamps.txt
            │   └── test_forward_flow_timestamps.csv

        '''

        num_bins = 32
        delta_t_ms =100

        self.seq_idx = seq_idx
        # Get Test Timestamp File
        # timestamp_file = seq_path / 'flow' / 'forward_timestamps.txt'
        # self.timestamps = np.loadtxt(timestamp_file, delimiter = ',', dtype='int64')

        test_timestamp_file = seq_path / 'test_forward_flow_timestamps.csv'
        assert test_timestamp_file.is_file()
        file = np.genfromtxt(
            test_timestamp_file,
            delimiter=','
        )
        self.idx_to_visualize = file[:,2]

        #Load and compute timestamps and indices
        timestamps_images = np.loadtxt(seq_path / 'image_timestamps.txt', dtype='int64')
        image_indices = np.arange(len(timestamps_images))
        # But only use every second one because we train at 10 Hz, and we leave away the 1st & last one
        self.timestamps_flow = timestamps_images[::2][1:-1]
        self.indices = image_indices[::2][1:-1]


        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins

        # Just for now, we always train with num_bins=15
        # assert self.num_bins==32

        self.voxel_grid = VoxelGrid((self.num_bins, self.height, self.width), normalize=True)

        # Save delta timestamp in ms
        self.delta_t_us = delta_t_ms * 1000


        # Left events only
        ev_dir_location = seq_path / 'events_left'
        ev_data_file = ev_dir_location / 'events.h5'
        ev_rect_file = ev_dir_location / 'rectify_map.h5'

        h5f_location = h5py.File(str(ev_data_file), 'r')
        self.h5f = h5f_location
        self.event_slicer = EventSlicer(h5f_location)
        with h5py.File(str(ev_rect_file), 'r') as h5_rect:
            self.rectify_ev_map = h5_rect['rectify_map'][()]

        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

    def events_to_voxel_grid(self, p, t, x, y, device: str='cpu'):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        # t= (t*0.00005).astype('float32')
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        event_data_torch = {
            'p': torch.from_numpy(pol),
            't': torch.from_numpy(t),
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
        }
        return self.voxel_grid.convert(event_data_torch)

    def getHeightAndWidth(self):
        return self.height, self.width

    @staticmethod
    def load_flow(flowfile: Path):
        assert flowfile.exists()
        assert flowfile.suffix == '.png'
        # print(str(flowfile))
        flow_16bit = imageio.imread(str(flowfile), format='PNG-FI')
        flow, valid2D = flow_16bit_to_float(flow_16bit)
        return flow, valid2D

    
    @staticmethod
    def close_callback(h5f):
        h5f.close()

    def get_image_width_height(self):
        return self.height, self.width

    def __len__(self):
        return len(self.timestamps_flow)-1

    def rectify_events(self, x: np.ndarray, y: np.ndarray):
        # assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_map
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def get_data_sample(self, index, crop_window=None, hflip=False, vflip=False):
        # First entry corresponds to all events BEFORE the flow map
        # Second entry corresponds to all events AFTER the flow map (corresponding to the actual fwd flow)
        # ts_start = [self.timestamps[index, 0], self.timestamps[index+1, 0]]
        # ts_end = [self.timestamps[index, 1], self.timestamps[index+1, 1]]

        ts_start = self.timestamps_flow[index]
        ts_end = self.timestamps_flow[index+1]
        
        file_index = self.indices[index]

        output = {
            'file_index': file_index,
            'timestamp': self.timestamps_flow[index]
        }
        # Save sample for benchmark submission
        output['save_submission'] = file_index in self.idx_to_visualize
        # output['visualize'] = self.visualize_samples


        event_data = self.event_slicer.get_events(ts_start, ts_end)

        p = event_data['p']
        t = event_data['t']
        x = event_data['x']
        y = event_data['y']

        xy_rect = self.rectify_events(x, y)
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]

        event_representation = self.events_to_voxel_grid(p, t, x_rect, y_rect)
        
        output['event_volume'] = event_representation

        return output

    def __getitem__(self, idx):
    
        sample =  self.get_data_sample(idx)        
            
        return sample

class DatasetProvider:
    def __init__(self, dataset_path: Path):
        assert dataset_path.is_dir(), str(dataset_path)

        self.seq_names = []
        sequences = list()
        idx = 0
        for child in dataset_path.iterdir():
            self.seq_names.append(str(child).split("/")[-1])
            sequences.append(Sequence(child, idx))
            idx += 1
        
        self.dataset = torch.utils.data.ConcatDataset(sequences)

    def get_dataset(self):
        return self.dataset

    def get_name_mapping(self):
        return self.seq_names

    def summary(self, logger):
        logger.write_line("================================== Dataloader Summary ====================================", True)
        logger.write_line("Loader Type:\t\t" + self.__class__.__name__, True)
