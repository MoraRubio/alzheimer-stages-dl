import numpy as np
import pandas as pd
from time import time, strftime
from sklearn.utils import class_weight

import torch
import torch.nn as nn

import monai
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, MapTransform, EnsureChannelFirstd, RandRotate90d, \
                             Resized, ScaleIntensityd, ToTensord, RandFlipd, RandZoomd
from monai.networks.nets import ViT, EfficientNetBN, DenseNet

from models.bilinear3D import Bilinear3D

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Working on device: {device}')

"""# Data"""

df = pd.read_csv('partition_tables/adni_table_3D_flirt_balanced_0.tsv', sep='\t')
#map_labels = {'CN':0, 'MCI':1, 'EMCI':2, 'AD':3, 'LMCI':3} # 4 classes
#map_labels = {'CN':0, 'MCI':1, 'EMCI':0, 'AD':1, 'LMCI':1} # 2 classes - Early vs Late
#df = df.loc[(df.Label=='CN') | (df.Label=='MCI')] # 2 classes
#map_labels = {'CN':0, 'MCI':1} # 2 classes
df = df.loc[(df.Label=='CN') | (df.Label=='MCI') | (df.Label=='AD')] # 3 classes
map_labels = {'CN':0, 'MCI':1, 'AD':2} # 3 classes
df['intLabel'] = df['Label'].map(map_labels)
n_classes=len(np.unique(df['intLabel'].values))
onehot = lambda x: torch.nn.functional.one_hot(torch.as_tensor(x), num_classes=n_classes).float()
df['onehot'] = df['intLabel'].apply(onehot)

groupby = df.groupby('Partition')
train_data = [{'img':img_path, 'lbl':label} for img_path, label in \
    zip(groupby.get_group('tr')['T1_path'].values, groupby.get_group('tr')['onehot'].values)]
print(f'Train images: {len(train_data)}')
val_data = [{'img':img_path, 'lbl':label} for img_path, label in \
    zip(groupby.get_group('dev')['T1_path'].values, groupby.get_group('dev')['onehot'].values)]
print(f'Validation images: {len(val_data)}')
test_data = [{'img':img_path, 'lbl':label} for img_path, label in \
    zip(groupby.get_group('te')['T1_path'].values, groupby.get_group('te')['onehot'].values)]
print(f'Test images: {len(test_data)}')

class LoadNPY(MapTransform):
    def __init__(self, keys, mode='valid'):
        MapTransform.__init__(self, keys)
        self.mode = mode
    def __call__(self, x):
        key = self.keys[0]
        data = x[key]
        x[key] = np.load(data, allow_pickle=True)[0]
        return x

train_transforms = Compose(
    [
        LoadNPY(keys=["img"]),
        EnsureChannelFirstd(keys=["img"], strict_check=False, channel_dim='no_channel'),
        ToTensord(keys=["img"]),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=(91, 91, 91)),
        RandRotate90d(keys=["img"], prob=0.2),
        RandFlipd(keys=["img"]),
        RandZoomd(keys=["img"]),
    ]
)
val_transforms = Compose(
    [
        LoadNPY(keys=["img"]),
        EnsureChannelFirstd(keys=["img"], strict_check=False, channel_dim='no_channel'),
        ToTensord(keys=["img"]),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=(91, 91, 91)),
    ]
)

train_ds = Dataset(data=train_data, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=1, pin_memory=pin_memory, drop_last=True)

val_ds = Dataset(data=val_data, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=1, pin_memory=pin_memory, drop_last=True)

"""# Training"""

class_weights = class_weight.compute_class_weight(class_weight='balanced', \
    classes=np.unique(df['intLabel'].values), y=df['intLabel'].values)

#model = Bilinear3D(n_classes=n_classes).to(device)
#model = DenseNet(spatial_dims=3, in_channels=1, out_channels=n_classes, dropout_prob=0.3).to(device)
model = EfficientNetBN(model_name="efficientnet-b7", pretrained=False, progress=False, \
                       spatial_dims=3, in_channels=1, num_classes=n_classes).to(device)
print(model)

loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))

optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-07)

val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
accuracy_values = []

log_flag = True
#experiment_name = f'Bilinear3D_{n_classes}Classes_CN_MCI_AD'
#experiment_name = f'DenseNet_{n_classes}Classes_CN_MCI_AD'
experiment_name = f'EfficientNet_{n_classes}Classes_CN_MCI_AD'

if log_flag:
    log = open(f"logs/log_{experiment_name}_{strftime('%d-%b-%Y-%H:%M:%S')}.csv", "w")
    log.write(f"Epochs,Train Loss,Train Accuracy,Val Accuracy,\n")

max_epochs = 250

for epoch in range(max_epochs):
    start_time = time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    train_num_correct = 0.0
    train_accuracy_count = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data['img'].to(device), batch_data['lbl'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs).as_tensor()
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len+1}, train_loss: {loss.item():.4f}")
        train_value = torch.eq(outputs.argmax(dim=1), labels.argmax(dim=1))
        train_accuracy_count += len(train_value)
        train_num_correct += train_value.sum().item()
    train_accuracy = train_num_correct / train_accuracy_count
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    print(f"epoch {epoch + 1} average accuracy: {train_accuracy:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        num_correct = 0.0
        accuracy_count = 0
        for val_data in val_loader:
            val_images, val_labels = val_data['img'].to(device), val_data['lbl'].to(device)
            with torch.no_grad():
                val_outputs = model(val_images).as_tensor()
                value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                accuracy_count += len(value)
                num_correct += value.sum().item()

        accuracy = num_correct / accuracy_count
        accuracy_values.append(accuracy)

        if accuracy > best_metric:
            best_metric = accuracy
            best_metric_epoch = epoch + 1
    
            if log_flag:
                torch.save(model.state_dict(), f"logs/best_acc_{experiment_name}.pth")
                print("Saved new best metric model")

        print(f"Current epoch: {epoch+1} current accuracy: {accuracy:.4f}") 
        print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")

        if log_flag:
            log.write(f"{epoch+1},{epoch_loss},{train_accuracy},{accuracy},\n")

    print(f"Epoch elapsed time: {time()-start_time:.4f}")

print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
if log_flag:
    log.close()