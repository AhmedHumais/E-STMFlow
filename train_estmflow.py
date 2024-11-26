    
from __future__ import print_function, division
from collections import OrderedDict
import sys
sys.path.append('core')

import argparse
import os
import numpy as np
from pathlib import Path
import json
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader
from model.emamba import EMambFlow

import pytorch_lightning as pl


from loader.loader_mamba import DatasetProvider
from pytorch_lightning.callbacks import ModelCheckpoint


# saves top-K checkpoints based on "val_loss" metric
checkpoint_callback = ModelCheckpoint(
    save_last=True,
    save_top_k=3,
    monitor="epe_val",
    mode="min"
)

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

class ESTMFlow_Trainer(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.model= EMambFlow(args)
        
    def load_checkpoint(self, checkpt):
        def load_model(mdl):
            mdl2 = OrderedDict()
            for key, v in mdl.items():
                mdl2[key.replace("model.","")] = v            
            return mdl2
        self.model.load_state_dict(load_model(checkpt), strict=True)
        
    def training_step(self, batch, batch_idx):

        flow = batch['flow']
        valid = batch['valid']

        flow_prediction, flow_new = self.model(batch['event_volume'])        

        loss, metrics = self.L1_loss(flow_prediction, flow, valid)
        
        metrics = {f'{key}_train': value for key, value in metrics.items()}
        self.log_dict(metrics, prog_bar=True, batch_size=1)

        return loss

    def validation_step(self, batch, batch_idx):

        flow = batch['flow']
        valid = batch['valid']

        flow_prediction, _ = self.model(batch['event_volume'])            

        loss, metrics = self.L1_loss(flow_prediction, flow, valid)
        metrics = {f'{key}_val': value for key, value in metrics.items()}
        self.log_dict(metrics, prog_bar=True, batch_size=1)
                        
        return loss
           
    def configure_optimizers(self):
        if self.args.ssm == "s4d" or self.args.ssm == "s4":
            print("using s4d/s4 specialized optimizer")
            optimizer, scheduler = self.setup_optimizer(lr=self.args.lr, weight_decay=self.args.wdecay, epochs=self.args.num_steps * 6000//(self.args.batch_size*self.args.batch_multiplier))
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay, eps=self.args.epsilon)
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, self.args.lr, epochs=self.args.num_steps, 
                                                    steps_per_epoch=6000//(self.args.batch_size*self.args.batch_multiplier) )

        # return optimizer
        return {    'optimizer': optimizer,
                    'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'} }
    
    def setup_optimizer(self, lr, weight_decay, epochs):
        """
        S4 requires a specific optimizer setup.

        The S4 layer (A, B, C, dt) parameters typically
        require a smaller learning rate (typically 0.001), with no weight decay.

        The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
        and weight decay (if desired).
        """

        # All parameters in the model
        all_parameters = list(self.model.parameters())

        # General parameters don't contain the special _optim key
        params = [p for p in all_parameters if not hasattr(p, "_optim")]

        # Create an optimizer with the general parameters
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **hp}
            )

        # Create a lr scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=epochs)

        # Print optimizer info
        keys = sorted(set([k for hp in hps for k in hp.keys()]))
        for i, g in enumerate(optimizer.param_groups):
            group_hps = {k: g.get(k, None) for k in keys}
            print(' | '.join([
                f"Optimizer group {i}",
                f"{len(g['params'])} tensors",
            ] + [f"{k} {v}" for k, v in group_hps.items()]))

        return optimizer, scheduler

    def L1_loss(self, flow_pred, flow_gt, valid, max_flow=MAX_FLOW):
        """ Loss function defined over sequence of flow predictions """

        valid = valid[:,1]           # because of padding issue of output of nn

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < max_flow)

        i_loss = (flow_pred - flow_gt).abs()
        flow_loss = (valid[:, None] * i_loss).mean()

        epe = torch.sum((flow_pred - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]
        if epe.nelement() == 0:
            epe = torch.zeros((1, 1))

        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
        }

        return flow_loss, metrics

def load_args(filename):
    args = argparse.Namespace()
    with open(filename, 'r') as f:
        args.__dict__ = json.load(f)
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='m_x2', help="config file name")
    
    cli_args = parser.parse_args()
    
    args = load_args(f'configs/{cli_args.config}.json')
    
    # torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    n_workers = 10

    data_path = Path('DSEC_Dataset/train')
    with open('train_split.txt', 'r') as f:
        train_seqs = f.read()
    with open('val_split.txt', 'r') as f:
        val_seqs = f.read()

    data_provider = DatasetProvider(data_path, train_seqs.split(), n_bins=args.n_bins, no_aug=False)
    val_provider = DatasetProvider(data_path, val_seqs.split(), n_bins=args.n_bins, no_aug=True)
    train_set = data_provider.get_dataset()
    val_set = val_provider.get_dataset()

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=n_workers, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=n_workers, pin_memory=True, shuffle=False)
    
    trainer = pl.Trainer(
        accelerator="gpu", 
        max_epochs=args.num_steps, 
        devices= args.gpus,
        gradient_clip_val=args.clip, 
        limit_train_batches=1.0,
        limit_val_batches=1.0, 
        default_root_dir="checkpoints", 
        accumulate_grad_batches=args.batch_multiplier,
        logger=True, 
        callbacks=[checkpoint_callback]
        )
    
    module = ESTMFlow_Trainer(args)

    trainer.fit(module, train_loader, val_loader, ckpt_path=args.ckpt) 
