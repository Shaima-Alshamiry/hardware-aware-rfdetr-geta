# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""RF-DETR training package (PyTorch Lightning).

Provides the Lightning module, data module, callbacks, and CLI for
training and evaluation.

Exports:
    RFDETRModelModule: LightningModule wrapping the RF-DETR model and training loop.
    RFDETRDataModule: LightningDataModule wrapping dataset construction and loaders.
    build_trainer: Factory that assembles a PTL Trainer from RF-DETR configs.
"""

from pytorch_lightning import seed_everything

from geta.tutorials.src.rfdetr.training.callbacks import (
    BestModelCallback,
    COCOEvalCallback,
    DropPathCallback,
    RFDETREarlyStopping,
    RFDETREMACallback,
)
from geta.tutorials.src.rfdetr.training.checkpoint import convert_legacy_checkpoint
from geta.tutorials.src.rfdetr.training.cli import RFDETRCli
from geta.tutorials.src.rfdetr.training.module_data import RFDETRDataModule
from geta.tutorials.src.rfdetr.training.module_model import RFDETRModelModule
from geta.tutorials.src.rfdetr.training.trainer import build_trainer
from geta.tutorials.src.rfdetr.utilities.logger import get_logger

_logger = get_logger()

__all__ = [
    "BestModelCallback",
    "COCOEvalCallback",
    "DropPathCallback",
    "RFDETRCli",
    "RFDETRDataModule",
    "RFDETREMACallback",
    "RFDETREarlyStopping",
    "RFDETRModelModule",
    "build_trainer",
    "convert_legacy_checkpoint",
    "seed_everything",
]
