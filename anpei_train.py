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
from make_it_dense.models import VoxelCompletionNet, RefineCompletionNet
from make_it_dense.utils import load_config

def train(
    config_file: Path = typer.Option(Path("./kitti_voxel.yaml"), "--config", "-c", exists=True),
    overfit_batches: int = 0,
    overfit_sequence: str = "",
    name: str = ""):
    
    config = load_config(config_file)

    '''
    note-0609

        add voxel-point-based coarse-to-fine strategy
        
        stage-1: train voxel-based u-net
        stage-2: train point-based u-net
    '''
    # == stage-1 voxel-based scene completion == #
    #    (16x/8x/4x/2x)
    # stage-1-a focus on global scene completion
    #           occ_th = 0.25 (inference)
    #           occ_th = 0.25 (training)
    config.is_train_global = True
    model = VoxelCompletionNet(config)
    model = VoxelCompletionNet.load_from_checkpoint(
        checkpoint_path=str("./models/pre-global-scene.ckpt"), config=config)
    data   = KittiDenseVoxelModule(config)    

    # stage-1-b focus on local object completion
    #           occ_th = 0.36 (inference)
    #           occ_th = 0.25 (training)
    # config.is_train_global = False 
    # model = VoxelCompletionNet(config)
    # model = VoxelCompletionNet.load_from_checkpoint(
    #     checkpoint_path=str("./models/pre-local-object.ckpt"), config=config)
    # data   = KittiDenseVoxelModule(config)

    # == stage-2 voxel-based scene completion == #
    #    (1x)
    # stage-2-a focus on global scene completion
    #           occ_th = 0.25 (inference)
    #           occ_th = 0.25 (training)
    # config.is_train_global = True
    # model = RefineCompletionNet(config)
    # model.add_pretrain_model(
    #     path="./models/pre-global-scene.ckpt", config=config)
    # data  = KittiDenseVoxelModule(config)
    # model = RefineCompletionNet.load_from_checkpoint(
    #     checkpoint_path=str("./models/global-scene.ckpt"), config=config)

    # stage-2-b focus on local object completion
    #           occ_th = 0.36 (inference)
    #           occ_th = 0.25 (training)
    # config.is_train_global = False 
    # model = RefineCompletionNet(config)
    # model.add_pretrain_model(
    #     path="./models/pre-local-object.ckpt", config=config)
    # data  = KittiDenseVoxelModule(config)
    # model = RefineCompletionNet.load_from_checkpoint(
    #     checkpoint_path=str("./models/a.ckpt"), config=config)

    # == stage-3 local and global fusion == #
    #    (local + global)
    # model = RefineCompletionNet(config)
    # model.add_pretrain_model_global(
    #     path="./models/pre-global-scene.ckpt", config=config)
    # model.add_pretrain_model_local(
    #     path="./models/pre-local-object.ckpt", config=config)
    # model = RefineCompletionNet.load_from_checkpoint(
    #     checkpoint_path=str("./models/a.ckpt"), config=config)
    # data   = KittiDenseVoxelModule(config)

    torch.multiprocessing.set_sharing_strategy(config.settings.sharing_strategy)

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=config.logging.save_dir,
        name=config.logging.name + "_" + name if name else config.logging.name,
        log_graph=config.logging.log_graph,
        default_hp_metric=False,
    )

    lr_monitor_callback = pl.callbacks.LearningRateMonitor(
        logging_interval=config.logging.lr_monitor_step,
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=config.checkpoints.monitor,
        save_top_k=config.checkpoints.save_top_k,
        mode=config.checkpoints.mode,
        save_on_train_epoch_end=True,
    )

    '''
    note-0511
        in the debuge mode (cpu), set gpus=0
    note-0518
        add precision=16 to decrease gpu memory (not good)
    '''
    trainer = pl.Trainer(
        default_root_dir="../",
        gpus=1 if config.settings.gpu else 0,
        max_epochs=config.training.n_epochs,
        overfit_batches=overfit_batches,
        logger=tb_logger,
        log_every_n_steps=config.logging.log_every_n_steps,
        weights_summary=config.logging.weights_summary,
        callbacks=[checkpoint_callback, lr_monitor_callback],
        auto_scale_batch_size="power",
        precision=16
    )
    trainer.fit(model, data)

if __name__ == "__main__":
    typer.run(train)