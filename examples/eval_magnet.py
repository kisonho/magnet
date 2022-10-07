"""
Main testing script to test a MAGNET on Cardiac dataset
"""
import os, torch
from magnet import MAGNET, Manager
from monai.data.dataloader import DataLoader
from monai.data.utils import pad_list_data_collate
from monai.metrics.meandice import DiceMetric
from torchmanager_monai import metrics

import data
from configs import EvaluationConfig

if __name__ == "__main__":
    # get configs
    config = EvaluationConfig.from_arguments()
    config.show_settings()

    # initialize data pathes and transform options
    image_dir = os.path.join(config.data, "Images.Testing")
    label_dir = os.path.join(config.data, "Labels.Testing")
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

    # load dataset
    _, testing_dataset, _, _ = data.load(image_dir, label_dir, config.img_size, train_split=24, transform_options=td)
    testing_dataset = DataLoader(testing_dataset, batch_size=1, collate_fn=pad_list_data_collate, pin_memory=True, num_workers=12)

    # load checkpoint
    if config.model.endswith(".model"):
        manager = Manager.from_checkpoint(config.model)
        if not isinstance(manager, Manager): raise TypeError("The manager is not in correct type.")
        model = manager.model.module if isinstance(manager.model, torch.nn.parallel.DataParallel) else manager.model
        print(f"Loaded checkpoint at epoch {manager.current_epoch}.")
        assert isinstance(model, MAGNET), "The model in checkpoint is not a valid `MAGNET` model."
    else:
        model = torch.load(config.model)
        dice_fn = metrics.CumulativeIterationMetric(DiceMetric(include_background=True, reduction="none", get_not_nans=False), target="out")
        metric_fns: dict[str, metrics.Metric] = {"dice": dice_fn}
        manager = Manager(model, metrics=metric_fns)
        print(f"Loaded saved model.")
        assert isinstance(model, MAGNET), "The model in checkpoint is not a valid `MAGNET` model."

    # set target modality
    if config.modality is not None: manager.target = config.modality # type: ignore
    else: manager.target = 0

    # evaluation
    assert isinstance(manager, Manager) and isinstance(testing_dataset, DataLoader)
    results = manager.test(testing_dataset, device=config.device, show_verbose=config.show_verbose)
    print(results)