"""
Main testing script to test a UNETR on iSEG dataset
"""
import torch
from monai.data.dataloader import DataLoader
from monai.data.utils import pad_list_data_collate
from monai.metrics.meandice import DiceMetric
from torchmanager_monai import metrics, Manager

import data
from configs import EvaluationConfig

if __name__ == "__main__":
    # get configs
    config = EvaluationConfig.from_arguments()
    config.show_settings()

    # initialize data pathes and transform options
    td = data.TransformOptions(
        load_imaged=False,
        center_spatial_cropd=(128, 128, 128),
        convert_label=True,
        set_modality=config.modality,
        orientationd=True,
        rand_crop_by_pos_neg_labeld=4,
        rand_flipd=True,
        rand_rotate_90d=True,
        rand_shift_intensityd=True,
        to_tensord=True
    )

    # load dataset
    _, _, testing_dataset, _, _ = data.load_iseg2017(config.data, config.img_size, td)
    testing_dataset = DataLoader(testing_dataset, batch_size=1, collate_fn=pad_list_data_collate, pin_memory=True, num_workers=12)

    # load checkpoint
    if config.model.endswith(".model"):
        manager = Manager.from_checkpoint(config.model)
        if not isinstance(manager, Manager): raise TypeError("The manager is not in correct type.")
        model = manager.model.module if isinstance(manager.model, torch.nn.parallel.DataParallel) else manager.model
        print(f"Loaded checkpoint at epoch {manager.current_epoch}.")
    else:
        model = torch.load(config.model)
        dice_fn = metrics.CumulativeIterationMetric(DiceMetric(include_background=True, reduction="none", get_not_nans=False), target="out")
        metric_fns: dict[str, metrics.Metric] = {"dice": dice_fn}
        manager = Manager(model, metrics=metric_fns)
        print(f"Loaded saved model.")

    # evaluation
    assert isinstance(manager, Manager) and isinstance(testing_dataset, DataLoader)
    results = manager.test(testing_dataset, device=config.device, show_verbose=config.show_verbose)
    print(results)