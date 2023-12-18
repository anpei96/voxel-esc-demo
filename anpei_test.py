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

def test(
    config_file: Path = typer.Option(Path("./kitti_voxel.yaml"), "--config", "-c", exists=True),
    overfit_batches: int = 0,
    overfit_sequence: str = "",
    name: str = ""):

    config = load_config(config_file)

    # == stage-2 voxel-based scene completion == #
    #    (1x)
    
    # stage-2-a focus on global scene completion
    #           occ_th = 0.25 (inference)
    #           occ_th = 0.25 (training)
    # ps: when inference again, remember to fuse local and global models
    config.is_train_global = True
    model = RefineCompletionNet(config)
    model.add_pretrain_model(
        path="./models/pre-global-scene.ckpt", config=config)
    data  = KittiDenseVoxelModule(config)
    model = RefineCompletionNet.load_from_checkpoint(
        checkpoint_path=str("./models/global-scene.ckpt"), config=config)
    
    # stage-2-b focus on local object completion
    #           occ_th = 0.36 (inference)
    #           occ_th = 0.25 (training)
    # config.is_train_global = False 
    # model = RefineCompletionNet(config)
    # model.add_pretrain_model(
    #     path="./models/pre-local-object.ckpt", config=config)
    # data  = KittiDenseVoxelModule(config)
    # model = RefineCompletionNet.load_from_checkpoint(
    #     checkpoint_path=str("./models/local-object.ckpt"), config=config)
    

    model.eval()
    torch.multiprocessing.set_sharing_strategy(config.settings.sharing_strategy)

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=config.logging.save_dir,
        name=config.logging.name + "_" + name if name else config.logging.name,
        log_graph=config.logging.log_graph,
        default_hp_metric=False,
    )

    trainer = pl.Trainer(
        default_root_dir="../",
        gpus=1 if config.settings.gpu else 0,
        max_epochs=config.training.n_epochs,
        overfit_batches=overfit_batches,
        logger=tb_logger,
        log_every_n_steps=config.logging.log_every_n_steps,
        weights_summary=config.logging.weights_summary,
        auto_scale_batch_size="power",
    )

    # ==> used for testing
    # re-define dataset sequence id
    # data.reset_test_set(eval_seq="00")
    # data.reset_test_set(eval_seq="01")
    # data.reset_test_set(eval_seq="02")
    # data.reset_test_set(eval_seq="03")
    # data.reset_test_set(eval_seq="04")
    # data.reset_test_set(eval_seq="05")
    # data.reset_test_set(eval_seq="06")
    # data.reset_test_set(eval_seq="08")
    # data.reset_test_set(eval_seq="09")
    # data.reset_test_set(eval_seq="10")
    
    # == used for validation
    data.reset_test_set(eval_seq="07")

    trainer.test(model, data)

if __name__ == "__main__":
    typer.run(test)