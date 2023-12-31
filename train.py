#!/usr/bin/env python
from pathlib import Path

import pytorch_lightning as pl
import torch
import typer

from make_it_dense.dataset import KITTIVDBDataModule
from make_it_dense.models import CompletionNet
from make_it_dense.utils import load_config


def train(
    config_file: Path = typer.Option(Path("config/kitti.yaml"), "--config", "-c", exists=True),
    overfit_batches: int = 0,
    overfit_sequence: str = "",
    name: str = "",
):
    config = load_config(config_file)
    torch.multiprocessing.set_sharing_strategy(config.settings.sharing_strategy)
    model = CompletionNet(config)
    data = KITTIVDBDataModule(config, overfit_sequence)
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

    # add precision=16
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
        #precision=16
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    typer.run(train)
