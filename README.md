# MAGNET: A Modality-Agnostic Networks for Medical Image Segmentation
Accepted by *ISBI 2023* \
Notice: This version contains improved features but will not affect the results. For original code used in paper, check the original [v1.0](https://github.com/kisonho/magnet/tree/stable-1.0)

## Pre-request
* Python >= 3.9
* PyTorch >= 1.12.1
* [torchmanager](https://github.com/kisonho/torchmanager) >= 1.0.4
* Monai >= 0.9.0

## Get Started
1. Convert multiple datasets to a `magnet.data.TargetDataset` and use `magnet.data.TargetedDataLoader` to load the data
```
import ...
import magnet

# load data
train_dataset_1, val_dataset_1 = ...
train_dataset_2, val_dataset_2 = ...
...
target_dict = {0: "m1", 1: "m2", ...}
training_dataset = magnet.data.TargetedDataset(train_dataset_1, train_dataset_2, target_dict=target_dict)
training_dataset = magnet.data.TargetedDataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
validation_dataset = {
    "m1": data.DataLoader(val_dataset_1, batch_size=batch_size, shuffle=False),
    "m2": data.DataLoader(val_dataset_2, batch_size=batch_size, shuffle=False),
	...
}
```

2. Simpy build the MAGNET (UNETR backbone) with `magnet.build` function
```
num_modalities: int = ...
num_classes: int = ...
img_size: Union[int, Sequence[int]] = ...
model = magnet.build(num_modalities, num_classes, img_size, target_dict=target_dict)
```

3. Or use the deeper `magnet.nn` framework to share layers in a `torch.nn.Module` by given names manually
```
model1: torch.nn.Module = ...
model2: torch.nn.Module = ...
shared_modules = {
	"layer1": model_to_share.layer1,
	"layer2": model_to_share.layer2,
	...
}
model = magnet.nn.share_modules([model1.some_layers, model2.some_layers], shared_modules, target_dict=target_dict)
```

4. Compile manager and train/test
```
optimizer = ...
loss_fn = ...
metric_fns = ...

epochs = ...
callbacks = ...

manager = magnet.Manager(model, optimizer, loss_fn=loss_fn, metric_fns=metric_fns)
manager.fit(training_dataset, epochs, val_dataset=validation_dataset, callbacks=callbacks)
summary.test(validation_dataset)
print(summary)
```

## Monai Support
* Using `magnet.MonaigManager` instead of `Manager` 
* Post processing support with `post_labels` and `post_predicts`
```
post_labels = [...]
post_predicts = [...]

manager = magnet.MonaigManager(model, post_labels=post_labels, post_predicts=post_predicts, optimizer=optimizer, loss_fn=loss_fn, metric_fns=metric_fns)
```
