"""
Main training script to train a UNETR
"""
import logging, os, torch
from magnet.networks import UNETRWithDictOutput as UNETR
from monai.data.dataloader import DataLoader
from monai.data.utils import pad_list_data_collate
from monai.losses.dice import DiceCELoss
from monai.metrics.meandice import DiceMetric
from torch.backends import cudnn
from torchmanager import callbacks, losses
from torchmanager_monai import Manager, metrics

import data
from configs import TrainingConfig

if __name__ == "__main__":
    # get configurations
    config = TrainingConfig.from_arguments()
    cudnn.benchmark = True
    if config.show_verbose: config.show_settings()

    # initialize checkpoint and data dirs
    data_dir = os.path.join(config.experiment_dir, "data")
    best_ckpt_dir = os.path.join(config.experiment_dir, "best.model")
    last_ckpt_dir = os.path.join(config.experiment_dir, "last.model")

    td = data.TransformOptions(
        load_imaged=True,
        add_channeld=False,
        convert_label=False,
        spacingd=None,
        set_modality=config.modality,
        normalize_intensityd=False,
        norm_to_one=False,
        crop_structure=None,
        spatial_padd=False,
        orientationd=True,
        rand_crop_by_pos_neg_labeld=4,
        rand_flipd=True,
        rand_rotate_90d=True,
        rand_shift_intensityd=True,
        to_tensord=True
        )

    # initialize dataset dir
    image_dir = os.path.join(config.data, "Images.Training")
    label_dir = os.path.join(config.data, "Labels.Training")

    # load dataset
    training_dataset, validation_dataset, in_channels, num_classes = data.load(image_dir, label_dir, config.img_size, train_split=config.training_split, transform_options=td, show_verbose=config.show_verbose)
    training_dataset = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=pad_list_data_collate)
    validation_dataset = DataLoader(validation_dataset, batch_size=1, collate_fn=pad_list_data_collate)

    # load model
    model = UNETR(in_channels, num_classes, img_size=config.img_size, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="instance", res_block=True, dropout_rate=0.0) if config.pretrained_model is None else torch.load(config.pretrained_model)

    # initialize optimizer, loss, metrics, and post processing
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = losses.Loss(DiceCELoss(to_onehot_y=True, softmax=True), target="out")
    dice_fn = metrics.CumulativeIterationMetric(DiceMetric(include_background=True, reduction="none", get_not_nans=False), target="out")
    metric_fns: dict[str, metrics.Metric] = {"val_dice": dice_fn}
    post_labels = data.transforms.AsDiscrete(to_onehot=num_classes)
    post_predicts = data.transforms.AsDiscrete(argmax=True, to_onehot=num_classes)

    # compile manager
    manager = Manager(model, post_labels=post_labels, post_predicts=post_predicts, optimizer=optimizer, loss_fn=loss_fn, metrics=metric_fns, roi_size=config.img_size) # type: ignore

    # initialize callbacks
    tensorboard_callback = callbacks.TensorBoard(data_dir)
    last_ckpt_callback = callbacks.LastCheckpoint(manager, last_ckpt_dir)
    besti_ckpt_callback = callbacks.BestCheckpoint("dice", manager, best_ckpt_dir)
    callbacks_list: list[callbacks.Callback] = [tensorboard_callback, besti_ckpt_callback, last_ckpt_callback]

    # validate before finetuning
    if config.pretrained_model is not None:
        summary = manager.test(validation_dataset, config.device, use_multi_gpus=config.use_multi_gpus, show_verbose=config.show_verbose)
        logging.info(summary)

    # train
    manager.fit(training_dataset, config.epochs, val_dataset=validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, callbacks_list=callbacks_list, show_verbose=config.show_verbose)

    # save and test
    model = manager.model
    torch.save(model, config.output_model)
    summary = manager.test(validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, show_verbose=config.show_verbose)
    logging.info(summary)