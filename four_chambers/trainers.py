"""Shorter nnU-Net training schedules for a few dozen 2D images on a 6 GB GPU.

nnU-Net's default trainer runs 1000 epochs x 250 iterations, designed for large
3D datasets on a modern GPU. With ~50 annotated 2D images (cropped around the
heart) that is thousands of passes over each image, i.e. days on a GTX 1660.
These variants change ONLY the schedule (number of epochs, iterations per
epoch); everything else (network, loss, augmentation, poly learning rate) is
nnU-Net's default.

nnU-Net looks trainers up by class name inside its own package, so train.py
copies this file to nnunetv2/training/nnUNetTrainer/variants/fetal4ch/.
"""
import torch
from torch import autocast as _torch_autocast

import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as _base
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

ITERATIONS_PER_EPOCH = 50
VAL_ITERATIONS_PER_EPOCH = 10


def _fp32_autocast(device_type, enabled=True, **kwargs):
    return _torch_autocast(device_type, enabled=False, **kwargs)


class nnUNetTrainerFetal(nnUNetTrainer):
    """nnU-Net with a short schedule, trained in full precision (fp32).

    nnU-Net trains with mixed precision (fp16). On the GTX 1660 SUPER (Turing
    TU116, no tensor cores) this produced NaN losses from the first epoch,
    a known problem of fp16 on GTX 16xx cards, so autocast is disabled here.
    """
    EPOCHS = 100

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.EPOCHS
        self.num_iterations_per_epoch = ITERATIONS_PER_EPOCH
        self.num_val_iterations_per_epoch = VAL_ITERATIONS_PER_EPOCH
        # full precision: no autocast in train/validation steps, no gradient scaling
        _base.autocast = _fp32_autocast
        self.grad_scaler = None


class nnUNetTrainerFetal_1ep(nnUNetTrainerFetal):
    EPOCHS = 1


class nnUNetTrainerFetal_5ep(nnUNetTrainerFetal):
    EPOCHS = 5


class nnUNetTrainerFetal_50ep(nnUNetTrainerFetal):
    EPOCHS = 50


class nnUNetTrainerFetal_100ep(nnUNetTrainerFetal):
    EPOCHS = 100


class nnUNetTrainerFetal_150ep(nnUNetTrainerFetal):
    EPOCHS = 150


class nnUNetTrainerFetal_200ep(nnUNetTrainerFetal):
    EPOCHS = 200


class nnUNetTrainerFetal_300ep(nnUNetTrainerFetal):
    EPOCHS = 300
