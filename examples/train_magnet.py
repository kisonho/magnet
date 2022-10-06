"""
Main training script to train a multi modality UNETR on shared structure on Challenge dataset
"""
import logging, magnet, os, torch
from monai.losses.dice import DiceCELoss
from monai.metrics.meandice import DiceMetric
from monai.transforms.transform import Transform
from torch.backends import cudnn
from torchmanager import callbacks, losses
from torchmanager_monai import metrics

import data
from configs import TrainingConfig

if __name__ == "__main__":
    # get configurations
    config = TrainingConfig.from_arguments()
    cudnn.benchmark = True
    if config.show_verbose: config.show_settings()

    # initialize checkpoint and data dirs
    data_dir = os.path.join(config.experiment_dir, "data")
    best_mri_ckpt_dir = os.path.join(config.experiment_dir, "best_mri.model")
    best_ct_ckpt_dir = os.path.join(config.experiment_dir, "best_ct.model")
    last_ckpt_dir = os.path.join(config.experiment_dir, "last.model")

    # transform options
    td_ct = data.TransformOptions(
        load_imaged=True,
        add_channeld=False,
        convert_label=False,
        spacingd=None,
        set_modality=0,
        normalize_intensityd=False,
        norm_to_one=False,
        crop_structure=None,
        grid_patchd=None,
        spatial_padd=False,
        orientationd=True,
        rand_crop_by_pos_neg_labeld=4,
        rand_flipd=True,
        rand_rotate_90d=True,
        rand_shift_intensityd=True,
        to_tensord=True
        )
    td_mri = data.TransformOptions(
        load_imaged=True,
        add_channeld=False,
        convert_label=False,
        spacingd=None,
        set_modality=1,
        normalize_intensityd=False,
        norm_to_one=False,
        crop_structure=None,
        grid_patchd=None,
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
    mri_train_dataset, mri_val_dataset, _, mri_num_classes = data.load(image_dir, label_dir, config.img_size, train_split=config.training_split, transform_options=td_mri, show_verbose=config.show_verbose)
    ct_train_dataset, ct_val_dataset, _, ct_num_classes = data.load(image_dir, label_dir, config.img_size, train_split=config.training_split, transform_options=td_ct, show_verbose=config.show_verbose)
    assert mri_num_classes == ct_num_classes, "MRI and CT classes are not equal."
    num_classes = mri_num_classes
    training_dataset = magnet.data.TargetedDataset(mri_train_dataset, ct_train_dataset, target_dict={0: "MRI", 1: "CT"})
    training_dataset = magnet.data.TargetedDataLoader(training_dataset, batch_size=config.batch_size, collate_fn=data.pad_list_data_collate, shuffle=True)
    validation_dataset = {
        "MRI": data.DataLoader(mri_val_dataset, batch_size=config.batch_size, collate_fn=data.pad_list_data_collate, shuffle=False),
        "CT": data.DataLoader(ct_val_dataset, batch_size=config.batch_size, collate_fn=data.pad_list_data_collate, shuffle=False),
    }

    # load model
    model = magnet.load(2, num_classes=num_classes, img_size=config.img_size, target_dict={0: "MRI", 1: "CT"})

    # initialize optimizer, loss, metrics, and post processing
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = losses.Loss(DiceCELoss(to_onehot_y=True, softmax=True), target="out")
    dice_fn = metrics.CumulativeIterationMetric(DiceMetric(include_background=True, reduction="none", get_not_nans=False), target="out")
    metric_fns: dict[str, metrics.Metric] = {"val_dice": dice_fn}
    post_labels: list[Transform] = [data.transforms.AsDiscrete(to_onehot=n) for n in [mri_num_classes, ct_num_classes]]
    post_predicts: list[Transform] = [data.transforms.AsDiscrete(argmax=True, to_onehot=n) for n in [mri_num_classes, ct_num_classes]]

    # compile manager
    manager = magnet.MonaiManager(model, post_labels, post_predicts, optimizer=optimizer, loss_fn=loss_fn, metrics=metric_fns, roi_size=config.img_size)

    # initialize callbacks
    tensorboard_callback = callbacks.TensorBoard(data_dir)
    last_ckpt_callback = callbacks.LastCheckpoint(manager, last_ckpt_dir)
    best_mri_ckpt_callback = callbacks.BestCheckpoint("dice_MRI", manager, best_mri_ckpt_dir)
    best_ct_ckpt_callback = callbacks.BestCheckpoint("dice_CT", manager, best_ct_ckpt_dir)
    callbacks_list: list[callbacks.Callback] = [tensorboard_callback, best_mri_ckpt_callback, best_ct_ckpt_callback, last_ckpt_callback]

    # train
    manager.fit(training_dataset, config.epochs, val_dataset=validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, callbacks_list=callbacks_list, show_verbose=config.show_verbose)

    # save and test
    model = manager.model
    torch.save(model, config.output_model)
    summary = manager.test(validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, show_verbose=config.show_verbose)
    logging.info(summary)