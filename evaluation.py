import numpy as np
import pandas as pd

import torch
from torchmetrics import Specificity, Recall

from monai.data import DataLoader, Dataset
from monai.transforms import Compose, MapTransform, EnsureChannelFirstd, \
                             Resized, ScaleIntensityd, ToTensord

from models.siamese3D import Siamese3D
from monai.networks.nets import ViT, EfficientNetBN, DenseNet

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Working on device: {device}')

"""# Data"""
n_fold=0
df = pd.read_csv(f'tables/adni_table_3D_flirt_balanced_{n_fold}.tsv', sep='\t')
#map_labels = {'CN':0, 'MCI':1, 'EMCI':2, 'AD':3, 'LMCI':3} # 4 classes
#map_labels = {'CN':0, 'MCI':1, 'EMCI':0, 'AD':1, 'LMCI':1} # 2 classes - Early vs Late
df = df.loc[(df.Label=='CN') | (df.Label=='AD')] # 2 classes
map_labels = {'CN':0, 'AD':1} # 2 classes
#df = df.loc[(df.Label=='CN') | (df.Label=='MCI') | (df.Label=='AD')] # 3 classes
#map_labels = {'CN':0, 'MCI':1, 'AD':2} # 3 classes
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

test_transforms = Compose(
    [
        LoadNPY(keys=["img"]),
        EnsureChannelFirstd(keys=["img"], strict_check=False, channel_dim='no_channel'),
        ToTensord(keys=["img"]),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=(91, 91, 91)),
    ]
)

train_ds = Dataset(data=train_data, transform=test_transforms)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=1, pin_memory=pin_memory, drop_last=True)

val_ds = Dataset(data=val_data, transform=test_transforms)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=1, pin_memory=pin_memory, drop_last=True)

test_ds = Dataset(data=test_data, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=1, pin_memory=pin_memory, drop_last=True)

"""# Evaluation"""
weights = f"EfficientNet_2Classes_CN_AD.pth"
#model = Siamese3D(n_classes=n_classes).to(device)
#model = DenseNet(spatial_dims=3, in_channels=1, out_channels=n_classes, dropout_prob=0.3).to(device)
model = EfficientNetBN(model_name="efficientnet-b7", pretrained=False, progress=False, \
                       spatial_dims=3, in_channels=1, num_classes=n_classes).to(device)
model.load_state_dict(torch.load('logs/'+weights))

print("Loaded weights "+weights)

model.eval()

num_correct = 0.0
accuracy_count = 0
specificity_func = Specificity(average='macro', num_classes=n_classes).to(device)
specificity_count = 0.0
sensitivity_func = Recall(average='macro', num_classes=n_classes).to(device)
sensitivity_count = 0.0
for train_data in train_loader:
    train_images, train_labels = train_data['img'].to(device), train_data['lbl'].to(device)
    with torch.no_grad():
        train_outputs = model(train_images).as_tensor()
        value = torch.eq(train_outputs.argmax(dim=1), train_labels.argmax(dim=1))
        accuracy_count += len(value)
        num_correct += value.sum().item()
        specificity_count += specificity_func(train_outputs.argmax(dim=1), train_labels.argmax(dim=1))
        sensitivity_count += sensitivity_func(train_outputs.argmax(dim=1), train_labels.argmax(dim=1))
accuracy = num_correct / accuracy_count
specificity = specificity_count / len(train_loader)
sensitivity = sensitivity_count / len(train_loader)
print(f"Accuracy on train set: {accuracy:.4f}, Specificity on train set: {specificity:.4f}, Sensitivity on train set: {sensitivity:.4f}")

num_correct = 0.0
accuracy_count = 0
specificity_count = 0.0
sensitivity_count = 0.0
for val_data in val_loader:
    val_images, val_labels = val_data['img'].to(device), val_data['lbl'].to(device)
    with torch.no_grad():
        val_outputs = model(val_images).as_tensor()
        value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
        accuracy_count += len(value)
        num_correct += value.sum().item()
        specificity_count += specificity_func(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
        sensitivity_count += sensitivity_func(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
accuracy = num_correct / accuracy_count
specificity = specificity_count / len(val_loader)
sensitivity = sensitivity_count / len(val_loader)
print(f"Accuracy on val set: {accuracy:.4f}, Specificity on val set: {specificity:.4f}, Sensitivity on val set: {sensitivity:.4f}")

num_correct = 0.0
accuracy_count = 0
specificity_count = 0.0
sensitivity_count = 0.0
for test_data in test_loader:
    test_images, test_labels = test_data['img'].to(device), test_data['lbl'].to(device)
    with torch.no_grad():
        test_outputs = model(test_images).as_tensor()
        value = torch.eq(test_outputs.argmax(dim=1), test_labels.argmax(dim=1))
        accuracy_count += len(value)
        num_correct += value.sum().item()
        specificity_count += specificity_func(test_outputs.argmax(dim=1), test_labels.argmax(dim=1))
        sensitivity_count += sensitivity_func(test_outputs.argmax(dim=1), test_labels.argmax(dim=1))
accuracy = num_correct / accuracy_count
specificity = specificity_count / len(test_loader)
sensitivity = sensitivity_count / len(test_loader)
print(f"Accuracy on test set: {accuracy:.4f}, Specificity on test set: {specificity:.4f}, Sensitivity on test set: {sensitivity:.4f}")

