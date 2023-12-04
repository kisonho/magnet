import data, glob, os, torch
from magnet import MAGNET, Manager
from monai.data.dataloader import DataLoader
from monai.data.utils import pad_list_data_collate

import metrics
from configs import EvaluationConfig


if __name__ == "__main__":
    # get configs
    configs = EvaluationConfig.from_arguments()
    configs.show_settings()

    # load dataset
    testing_image_dir, testing_label_dir = None, None
    for file in sorted(glob.glob(os.path.join(configs.data, '*'))):
        if 'Image' in file and 'Testing' in file: testing_image_dir = file
        if 'Labels' in file and 'Testing' in file: testing_label_dir = file
    assert testing_image_dir is not None and testing_label_dir is not None
    _, test_ds = data.load_brats2018(
        train_image_dir=testing_image_dir, 
        train_label_dir=testing_label_dir, 
        train_split=0, 
        img_size=configs.img_size,
        cache_num=(10, 10),
        chached=True,
        show_verbose=True, 
        )
    testing_dataset = DataLoader(test_ds, batch_size=1, collate_fn=pad_list_data_collate)

    # load checkpoint
    if configs.model.endswith(".model"):
        manager = Manager.from_checkpoint(configs.model)
        if not isinstance(manager, Manager): raise TypeError("The manager is not in correct type.")
        model = manager.model.module if isinstance(manager.model, torch.nn.parallel.DataParallel) else manager.model
        print(f"Loaded checkpoint at epoch {manager.current_epoch}.")
        assert isinstance(model, MAGNET), "The model in checkpoint is not a valid `MAGNET` model."
    else:
        model = torch.load(configs.model)
        print(f"Loaded saved model.")
        assert isinstance(model, MAGNET), "The model in checkpoint is not a valid `MAGNET` model."
        manager = Manager(model)

    # initialize metrics
    metric_fns: dict[str, metrics.Metric] = {
        # "dice": dice_fn,
        'Whole_tumor': metrics.whole_tumor_dice,
        'Tumor_Core': metrics.tumor_core_dice,
        'Ehancing_tumor': metrics.enhancing_tumor_dice
        }
    manager.metric_fns.update(metric_fns)

    # set target modality
    manager.target = configs.modality

    # evaluation
    assert isinstance(manager, Manager) and isinstance(testing_dataset, DataLoader)
    results = manager.test(testing_dataset, device=configs.device, show_verbose=configs.show_verbose)
    print(results)
