"""
Main training script to train MAGNET on BraTS2018
python train_BraTS2018.py _ _ --show_verbose -b 2 -e 300 -exp MAGNET_BraTS2018_run4 --img_size 80 80 80

python train_BraTS2018.py _ _ --show_verbose -b 2 -e 300 -exp Unet_BraTS2018_RFNet_imgsize112_FixedArchitecture_newsplit_2 --img_size 112 112 112
"""
import os, torch
torch.multiprocessing.set_sharing_strategy('file_system')
# print(torch.multiprocessing.get_sharing_strategy())
from monai.data.utils import pad_list_data_collate
from monai.losses.dice import DiceCELoss
from monai.metrics.meandice import DiceMetric
from torch.backends import cudnn
from torchmanager import callbacks
from torchmanager_core.view import logger
from typing import Callable

import configs, data, magnet
from magnet import losses
from torchmanager_monai import metrics
from torchmanager_monai.data import DataLoader


if __name__ == "__main__":
    # get configurations
    TrainingConfig = configs.SharedTrainingConfig
    config = TrainingConfig.from_arguments()
    cudnn.benchmark = True
    if config.show_verbose: config.show_settings()

    # initialize data dirs
    data_dir = os.path.join(config.experiment_dir, "data")
    training_image_dir = os.path.join(config.data, "Images.Training")
    training_label_dir = os.path.join(config.data, "Labels.Training")
    num_classes = 4
    targs = ['T2-Fl', 'T1-W', 'T1-Gd', 'T2-W']
    target_dict = {i: targ for i, targ in enumerate(targs)}
    in_channels=len(targs)

    # initialize transform options
    num_workers = os.cpu_count()
    num_workers = 0 if num_workers is None else num_workers
    num_workers = round(num_workers * 0.8)
    default_device = config.device if isinstance(config.device, torch.device) else config.device[0]

    # load training and validation dataset
    img_size = config.img_size
    train_ds, val_ds = data.load_brats2018(
        train_image_dir = training_image_dir,
        train_label_dir = training_label_dir,
        train_split = 29,
        img_size = img_size,
        chached = True,
        cache_num=(10, 10),
        show_verbose=True,
        num_workers=num_workers,
        logger=logger,
    )
    training_dataset = DataLoader(train_ds, batch_size=config.batch_size, collate_fn=pad_list_data_collate, shuffle=True, num_workers=num_workers, pin_memory=True, pin_memory_device=f"{default_device.type}:{default_device.index}")
    validation_dataset = DataLoader(val_ds, batch_size=1, collate_fn=pad_list_data_collate, num_workers=num_workers, pin_memory=True, pin_memory_device=f"{default_device.type}:{default_device.index}")

    # Building MAGNET v2 on UNET
    model = magnet.build_v2_unet(in_channels=in_channels, num_classes=num_classes, target_dict=target_dict, return_features=True)

    # RFNet Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5, amsgrad=True) # Need a "Poly" learning rate

    # add backward clip hook to avoid NAN gradients
    gradients_hook_fn: Callable[[torch.Tensor], torch.Tensor] = lambda g: g.clamp(-1, 1).nan_to_num(0)
    for param in model.parameters():
        param.register_hook(gradients_hook_fn)

    # initialize losses
    dice_losses = [losses.Loss(DiceCELoss(to_onehot_y=True, softmax=True)) for _ in range(in_channels + 1)]
    kldiv_losses: list[losses.Loss] = [losses.PixelWiseKLDiv(softmax_temperature=config.temperature, weight=config.distillation_lambda) for _ in range(in_channels)]
    mse_losses = [losses.Loss(torch.nn.MSELoss(reduction='mean'), weight=config.distillation_gamma) for _ in range(in_channels)]
    loss_fn = losses.MAGMSLoss(dice_losses, distillation_loss=kldiv_losses, feature_losses=mse_losses)  # type: ignore

    # metrics
    dice_fn = metrics.CumulativeIterationMetric(DiceMetric(include_background=True, reduction="none", get_not_nans=False))

    ## Metrics WITH self distillation
    metric_fns: dict[str, metrics.Metric] = {"val_dice": dice_fn}
    post_labels = [data.transforms.AsDiscrete(to_onehot=num_classes) for _ in range(in_channels)]
    post_predicts = [data.transforms.AsDiscrete(argmax=True, to_onehot=num_classes) for _ in range(in_channels)]

    # compile manager
    manager = sharing.MonaiManager(model, post_labels=post_labels, post_predicts=post_predicts, optimizer=optimizer, loss_fn=loss_fn, metrics=metric_fns, roi_size=config.img_size, target_freq=config.target_frequency)  # type: ignore

    # initialize callbacks
    experiment_callback = callbacks.Experiment(config.experiment, manager, monitors=["dice"])
    
    # Learning rate scheduler + All callbacks
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: ( 1 - (epoch / config.epochs)) ** 0.9)
    lr_scheduler_callback = callbacks.LrSchedueler(lr_scheduler, tf_board_writer=experiment_callback.tensorboard.writer)  # type: ignore
    callbacks_list = [experiment_callback, lr_scheduler_callback]

    # train model
    model = manager.fit(training_dataset, config.epochs, val_dataset=validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, callbacks_list=callbacks_list, show_verbose=config.show_verbose)

    # save final model
    output_model = os.path.join('experiments', config.experiment, 'Final.model')
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    torch.save(model, output_model)
