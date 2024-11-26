import numpy as np

from tqdm import tqdm
import cv2

def prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=1.0):
    flow_x_interp = cv2.remap(x_flow,
                              x_indices,
                              y_indices,
                              cv2.INTER_NEAREST)
    
    flow_y_interp = cv2.remap(y_flow,
                              x_indices,
                              y_indices,
                              cv2.INTER_NEAREST)

    x_mask[flow_x_interp == 0] = False
    y_mask[flow_y_interp == 0] = False
        
    x_indices += flow_x_interp * scale_factor
    y_indices += flow_y_interp * scale_factor

    return

def estimate_corresponding_gt_flow(x_flow_in,
                                   y_flow_in):
    # Each gt flow at timestamp gt_timestamps[gt_iter] represents the displacement between
    # gt_iter and gt_iter+1.

    gt_dt = 100e-3
    x_flow = np.squeeze(x_flow_in[0, ...])
    y_flow = np.squeeze(y_flow_in[0, ...])
    
    x_indices, y_indices = np.meshgrid(np.arange(x_flow.shape[1]),
                                       np.arange(x_flow.shape[0]))
    x_indices = x_indices.astype(np.float32)
    y_indices = y_indices.astype(np.float32)

    orig_x_indices = np.copy(x_indices)
    orig_y_indices = np.copy(y_indices)

    # Mask keeps track of the points that leave the image, and zeros out the flow afterwards.
    x_mask = np.ones(x_indices.shape, dtype=bool)
    y_mask = np.ones(y_indices.shape, dtype=bool)

    scale_factor = 1
    total_dt = gt_dt

    prop_flow(x_flow, y_flow,
              x_indices, y_indices,
              x_mask, y_mask,
              scale_factor=scale_factor)

    final_dt = gt_dt
    total_dt += final_dt

    final_gt_dt = gt_dt
    
    x_flow = np.squeeze(x_flow_in[1, ...])
    y_flow = np.squeeze(y_flow_in[1, ...])

    scale_factor = final_dt / final_gt_dt
    
    prop_flow(x_flow, y_flow,
              x_indices, y_indices,
              x_mask, y_mask,
              scale_factor)
    
    x_shift = x_indices - orig_x_indices
    y_shift = y_indices - orig_y_indices
    x_shift[~x_mask] = 0
    y_shift[~y_mask] = 0
    
    return x_shift, y_shift
  
def get2flow_gt(flow0, flow1, valid0):
  flow = np.stack((flow0, flow1))  
  x_flow, y_flow =  estimate_corresponding_gt_flow(flow[...,0], flow[...,1])
  
  label = np.stack((x_flow, y_flow), axis=-1)
  valid = valid0 & ((x_flow != 0) | (y_flow !=0))
  
  return label, valid