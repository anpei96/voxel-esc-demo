#!/usr/bin/env python

#
# project: lidar point cloud completion in the 
#          large-scale scene
#          (based on make-it-dense)
# author:  anpei
# email:   anpei@wit.edu.cn
# data:    05.08.2022
# 

from pathlib import Path

import pytorch_lightning as pl
import torch
import typer

from make_it_dense.dataset import KittiDenseVoxelModule
from make_it_dense.models import VoxelCompletionNet
from make_it_dense.utils import load_config

def val(
    config_file: Path = typer.Option(Path("./kitti_voxel.yaml"), "--config", "-c", exists=True),
    overfit_batches: int = 0,
    overfit_sequence: str = "",
    name: str = ""):
    
    config = load_config(config_file)
    data   = KittiDenseVoxelModule(config)
    model = VoxelCompletionNet.load_from_checkpoint(
        checkpoint_path=str("./models/voxel-coarse-d.ckpt"), config=config)
    model.eval()
    torch.multiprocessing.set_sharing_strategy(config.settings.sharing_strategy)

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=config.logging.save_dir,
        name=config.logging.name + "_" + name if name else config.logging.name,
        log_graph=config.logging.log_graph,
        default_hp_metric=False,
    )

    '''
    note -0511
        in the debuge mode (cpu), set gpus=0
    '''
    trainer = pl.Trainer(
        default_root_dir="../",
        gpus=1 if config.settings.gpu else 0,
        max_epochs=config.training.n_epochs,
        overfit_batches=overfit_batches,
        logger=tb_logger,
        log_every_n_steps=config.logging.log_every_n_steps,
        weights_summary=config.logging.weights_summary,
        auto_scale_batch_size="power",
        precision=16
    )
    trainer.test(model, data)

if __name__ == "__main__":
    typer.run(val)