from pathlib import Path
import random
from typing import Dict, Tuple
import weakref
import math

from numba import jit
import h5py
import hdf5plugin
import torch
from torch.utils.data import Dataset
import numpy as np
import imageio.v2 as imageio
import os

from utils.dsec_utils import RepresentationType, VoxelGrid, flow_16bit_to_float
from .dsec_flow_utils import get2flow_gt

class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000
        # (2) t[ms_to_idx[ms] - 1] < ms*1000
        # ,where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        self.t_offset = int(h5f['t_offset'][()])
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx], dtype=np.uint64)
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size
        return events


    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]


class Sequence(Dataset):
    def __init__(self, seq_path: Path, seq_idx: int, num_bins = 32, no_aug=False):
        
        assert seq_path.is_dir()
        '''
        Directory Structure:

        DSEC_Dataset
        └── train
            ├── thun_00_a
            │   ├── events
            │   │   ├── left
            │   │   │   ├── events.h5
            │   │   │   └── rectify_map.h5
            │   │   │  
            │   ├── flow
            │   │   ├── forward_timestamps.txt
            │   │   ├── forward
            │   │   │   ├── 000134.png

        '''
        self.no_aug = no_aug
        delta_t_ms =100

        self.seq_idx = seq_idx
        # Get Test Timestamp File
        timestamp_file = seq_path / 'flow' / 'forward_timestamps.txt'
        self.timestamps = np.loadtxt(timestamp_file, delimiter = ',', dtype='int64')

        self.flow_maps_path = seq_path / 'flow' / 'forward'
        self.flow_maps_list = os.listdir(str(self.flow_maps_path))
        self.flow_maps_list.sort()

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins

        # Set event representation
        self.rand_crop = False
        self.crop_size = (288,384)
        if self.rand_crop:
            self.voxel_grid = VoxelGrid((num_bins,self.crop_size[0], self.crop_size[1]), normalize=True)  #### check if normalizing is better or not
        else:        
            self.voxel_grid = VoxelGrid((num_bins,self.height, self.width), normalize=True)

        # Save delta timestamp in ms
        self.delta_t_us = delta_t_ms * 1000


        # Left events only
        ev_dir_location = seq_path / 'events' / 'left'
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
        flow_16bit = imageio.imread(str(flowfile), format='PNG-FI')
        flow, valid2D = flow_16bit_to_float(flow_16bit)
        return flow, valid2D

    @staticmethod
    def close_callback(h5f):
        h5f.close()

    def get_image_width_height(self):
        return self.height, self.width

    def __len__(self):
        return len(self.timestamps)-2

    def rectify_events(self, x: np.ndarray, y: np.ndarray):
        # assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_map
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def get_data_sample(self, index, crop_window=None, hflip=False, vflip=False, extend_time=False):

        ts_start = self.timestamps[index+1, 0]
        if extend_time:
            ts_end = self.timestamps[index+1, 0] + 2*self.delta_t_us   
        else:         
            ts_end = self.timestamps[index+1, 0] + self.delta_t_us
        
        output = {
            'seq_index': self.seq_idx,
            'timestamp': self.timestamps[index+1]
        }
        output['init_seq'] = (index == 0)

        diff = self.timestamps[index+1, 0] - self.timestamps[index, 0]
        if diff > 101000:
            output['init_seq'] = True
        if extend_time:
            flow0, valid0 = self.load_flow(self.flow_maps_path / self.flow_maps_list[index+1])
            flow1, _ = self.load_flow(self.flow_maps_path / self.flow_maps_list[index+2])
            flow, valid = get2flow_gt(flow0, flow1, valid0)
        else:
            flow, valid = self.load_flow(self.flow_maps_path / self.flow_maps_list[index+1])

        if crop_window is not None:
            flow = flow[crop_window['start_y']:crop_window['start_y']+crop_window['crop_height'], crop_window['start_x']: crop_window['start_x']+crop_window['crop_width']]
            valid = valid[crop_window['start_y']:crop_window['start_y']+crop_window['crop_height'], crop_window['start_x']: crop_window['start_x']+crop_window['crop_width']]

        flow = torch.from_numpy(flow.transpose(2, 0, 1))
        valid = torch.from_numpy(valid.astype('float32'))
        if hflip:
            flow = flow.flip(-1)
            flow[0] = -1 * flow[0]
            valid = valid.flip(-1)
        if vflip:
            flow = flow.flip(-2)
            flow[1] = -1 * flow[1]
            valid = valid.flip(-2)
        output['flow'] = flow
        output['valid'] = torch.stack([valid]*2)

        event_data = self.event_slicer.get_events(ts_start, ts_end)

        p = event_data['p']
        t = event_data['t']
        x = event_data['x']
        y = event_data['y']

        xy_rect = self.rectify_events(x, y)
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]

        if crop_window is not None:
            # Cropping (+- 2 for safety reasons)
            x_mask = (x_rect >= crop_window['start_x']-2) & (x_rect < crop_window['start_x']+crop_window['crop_width']+2)
            y_mask = (y_rect >= crop_window['start_y']-2) & (y_rect < crop_window['start_y']+crop_window['crop_height']+2)
            mask_combined = x_mask & y_mask
            p = p[mask_combined]
            t = t[mask_combined]
            x_rect = x_rect[mask_combined]
            y_rect = y_rect[mask_combined]

        event_representation = self.events_to_voxel_grid(p, t, x_rect, y_rect)
        
        if hflip:
            event_representation = event_representation.flip(-1)
        if vflip:
            event_representation = event_representation.flip(-2)

        output['event_volume'] = event_representation

        return output

    def __getitem__(self, idx):
        if self.rand_crop:
            top = random.randint(0, self.height - self.crop_size[0])
            left = random.randint(0, self.width - self.crop_size[1])
            crop_wind = {
                'start_x': left,
                'start_y': top,
                'crop_height': self.crop_size[0],
                'crop_width': self.crop_size[1]
            }
            sample =  self.get_data_sample(idx, crop_wind, hflip=(torch.rand(1).item() < 0.5), vflip=(torch.rand(1).item() < 0.4))        
        else:
            if self.no_aug:
                sample =  self.get_data_sample(idx) 
            else:
                sample =  self.get_data_sample(idx, hflip=(torch.rand(1).item() < 0.5), vflip=(torch.rand(1).item() < 0.5), extend_time=(torch.rand(1).item() < 0.3))        
            
        return sample

class DatasetProvider:
    def __init__(self, dataset_path: Path, seq_names, n_bins=32, no_aug=False):
        assert dataset_path.is_dir(), str(dataset_path)

        self.seq_names = seq_names
        sequences = list()
        idx = 0
        for seq in seq_names:
            sequences.append(Sequence(dataset_path / seq, idx, num_bins=n_bins, no_aug=no_aug))
            idx += 1
        

        self.dataset = torch.utils.data.ConcatDataset(sequences)

    def get_dataset(self):
        return self.dataset

    def get_name_mapping(self):
        return self.seq_names

    def summary(self, logger):
        logger.write_line("================================== Dataloader Summary ====================================", True)
        logger.write_line("Loader Type:\t\t" + self.__class__.__name__, True)
