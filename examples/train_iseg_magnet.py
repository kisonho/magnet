"""
Main training script to train a multi modality UNETR with fuse inputs on Eric's dataset
"""
import logging, magnet, os, torch
from monai.data.dataloader import DataLoader
from monai.data.utils import pad_list_data_collate
from monai.losses.dice import DiceCELoss
from monai.metrics.meandice import DiceMetric
from torch.backends import cudnn
from torchmanager import callbacks, losses

import data
from configs import TrainingConfig
from torchmanager_monai import metrics

if __name__ == "__main__":
    # get configurations
    config = TrainingConfig.from_arguments()
    cudnn.benchmark = True
    if config.show_verbose: config.show_settings()

    # initialize checkpoint and data dirs
    data_dir = os.path.join(config.experiment_dir, "data")
    best_t1_ckpt_dir = os.path.join(config.experiment_dir, "best_t1.model")
    best_t2_ckpt_dir = os.path.join(config.experiment_dir, "best_t2.model")
    last_ckpt_dir = os.path.join(config.experiment_dir, "last.model")

    # initialize transform options
    # transform options
    td = data.TransformOptions(
        load_imaged=False,
        center_spatial_cropd=(128, 128, 128),
        convert_label=True,
        set_modality=None,
        orientationd=True,
        rand_crop_by_pos_neg_labeld=4,
        rand_flipd=True,
        rand_rotate_90d=True,
        rand_shift_intensityd=True,
        to_tensord=True
        )
    td_t1 = data.TransformOptions(
        load_imaged=False,
        center_spatial_cropd=(128, 128, 128),
        convert_label=True,
        set_modality=0,
        orientationd=True,
        rand_crop_by_pos_neg_labeld=4,
        rand_flipd=True,
        rand_rotate_90d=True,
        rand_shift_intensityd=True,
        to_tensord=True
        )
    td_t2 = data.TransformOptions(
        load_imaged=False,
        center_spatial_cropd=(128, 128, 128),
        convert_label=True,
        set_modality=1,
        orientationd=True,
        rand_crop_by_pos_neg_labeld=4,
        rand_flipd=True,
        rand_rotate_90d=True,
        rand_shift_intensityd=True,
        to_tensord=True
        )

    # load dataset
    t1_training_dataset, t1_validation_dataset, t1_testing_dataset, in_channels, num_classes = data.load_iseg2017(config.data, config.img_size, td_t1, split=[7 - config.training_split, config.training_split, 3])
    t2_training_dataset, t2_validation_dataset, t2_testing_dataset, in_channels, num_classes = data.load_iseg2017(config.data, config.img_size, td_t2, split=[7 - config.training_split, config.training_split, 3])
    training_dataset = magnet.data.TargetedDataset(t1_training_dataset, t2_training_dataset, target_dict={0: "T1", 1: "T2"})
    training_dataset = magnet.data.TargetedDataLoader(training_dataset, batch_size=config.batch_size, collate_fn=data.pad_list_data_collate, shuffle=True)
    validation_dataset = {
        "T1": DataLoader(t1_validation_dataset, batch_size=1, collate_fn=pad_list_data_collate),
        "T2": DataLoader(t2_validation_dataset, batch_size=1, collate_fn=pad_list_data_collate),
    }

    # load model
    model = magnet.load(2, num_classes, config.img_size, target_dict={0: "T1", 1: "T2"})

    '''
    # load model
    UNETR = networks.TargetedUNETRWithDictOutput
    model = UNETR(2, num_classes, img_size=config.img_size, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="instance", res_block=True, dropout_rate=0.0, target_dict={0: "T1", 1: "T2"})
    '''

    # initialize optimizer, loss, metrics, and post processing
    optimizer = torch.optim.SGD(model.parameters(), 0.001, momentum=0.6)
    loss_fn = losses.Loss(DiceCELoss(to_onehot_y=True, softmax=True), target="out")
    dice_fn = metrics.CumulativeIterationMetric(DiceMetric(include_background=True, reduction="none", get_not_nans=False), target="out")
    metric_fns: dict[str, metrics.Metric] = {"val_dice": dice_fn}
    post_labels = [data.transforms.AsDiscrete(to_onehot=num_classes) for _ in range(2)]
    post_predicts = [data.transforms.AsDiscrete(argmax=True, to_onehot=num_classes) for _ in range(2)]

    # initialize learning rate scheduler
    lr_step = max(int(config.epochs / 6), 1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, gamma=0.5)

    # compile manager
    manager = magnet.Manager(model, post_labels=post_labels, post_predicts=post_predicts, optimizer=optimizer, loss_fn=loss_fn, metrics=metric_fns, roi_size=config.img_size, target_freq=config.target_frequency) # type: ignore

    # initialize callbacks
    tensorboard_callback = callbacks.TensorBoard(data_dir)
    last_ckpt_callback = callbacks.LastCheckpoint(manager, last_ckpt_dir)
    best_t1_ckpt_callback = callbacks.BestCheckpoint("dice_T1", manager, best_t1_ckpt_dir)
    best_t2_ckpt_callback = callbacks.BestCheckpoint("dice_T2", manager, best_t2_ckpt_dir)
    lr_scheduler_callback = callbacks.LrSchedueler(lr_scheduler, tf_board_writer=tensorboard_callback.writer)
    callbacks_list: list[callbacks.Callback] = [tensorboard_callback, best_t1_ckpt_callback, best_t2_ckpt_callback, last_ckpt_callback, lr_scheduler_callback]

    # train
    manager.fit(training_dataset, config.epochs, val_dataset=validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, callbacks_list=callbacks_list, show_verbose=config.show_verbose)

    # save and test
    model = manager.model
    os.makedirs(os.path.dirname(config.output_model), exist_ok=True)
    torch.save(model, config.output_model)
    summary = manager.test(validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, show_verbose=config.show_verbose)
    logging.info(summary)