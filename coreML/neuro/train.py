import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List

import utils as u
import models
import DatasetSpectrogram as ds

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def set_seed(random_state: int) -> int:
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    return random_state


def exchange(x: List[str], name: str = 'No_Healthy') -> int:
    if x == name:
        return 0
    else: return 1


def create_df(path: str, drop_list: List[str] = ['Unnamed: 0']) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(drop_list, axis=1)
    df.disease = list(map(exchange, df.disease.tolist()))
    return df


def train(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, optimizer: optim.Optimizer, 
            criterion: nn.Module, device: torch.device, epochs: int = 20, log_interval: int = 60) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    model.train()
    train_loss = []
    validation_loss = []
    for epoch in range(epochs):
        val_loss = 0
        tr_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device).float(), target.to(device).float()
            optimizer.zero_grad()
            output = model(data)
            #target = target.unsqueeze(1)
            loss = criterion(output, target)
            tr_loss += loss
            loss.backward()
            optimizer.step()
            if (batch_idx % log_interval == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data))
        with torch.no_grad():
            for data, target in validation_loader:
                data, target = data.to(device).float(), target.to(device).float()
                output = model(data)
                #target = target.unsqueeze(1)
                val_loss += criterion(output, target)
                pred = torch.round(torch.sigmoid(output))
                correct += (pred == target).sum()
        print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            val_loss / len(validation_loader.dataset), correct, len(validation_loader.dataset),
            100. * correct / len(validation_loader.dataset)))
        validation_loss.append(val_loss.cpu() / len(validation_loader.dataset))
        train_loss.append((tr_loss.cpu() / len(train_loader.dataset)).detach().numpy())
    return validation_loss, train_loss


def test(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device).float()
            output = model(data)
            #target = target.unsqueeze(1)
            test_loss += criterion(output, target).item()
            pred = torch.round(torch.sigmoid(output))
            correct += (pred == target).sum()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)
    

def ploting(model_name: str, spectrogram: str, validation_loss: float, train_loss: float, acc: float) -> None:
    plt.plot(validation_loss, label="Validation Loss", color="red")
    plt.plot(train_loss, label="Train Loss", color="blue")
    plt.legend(loc="upper right")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.savefig('save/{}_{}_{:.2f}.png'.format(model_name, spectrogram, acc))


parser = argparse.ArgumentParser(description='Train model script')
"""
- random_state
- path_train
- path_val
- path_test
- path_audio
- batch_size
- epochs
- spectrogram
- model
"""
parser.add_argument("-rs", dest='random_state', default=27, type=int)
parser.add_argument("-path_train", dest="path_train", default='df_train.csv', required=False)
parser.add_argument("-path_val", dest='path_val', default='df_val.csv', required=False)
parser.add_argument("-path_test", dest='path_test', default='df_test.csv', required=False)
parser.add_argument("-path_audio", dest='path_audio', default='preprocessed_audio', required=False)
parser.add_argument("-bs", dest='batch_size', default=4, type=int)
parser.add_argument("-epochs", dest='epochs', default=20, type=int)
parser.add_argument("-spec", dest='spectrogram', choices=['spec'])
parser.add_argument("-m", dest='model', required=True)


if __name__ == '__main__':
    args = parser.parse_args()
    random_state = set_seed(args.random_state)
    audio2 = u.SPECTROGRAM_MAP[args.spectrogram]

    if torch.cuda.is_available():
        print("Let's use", torch.cuda.device_count(), "GPUs")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = models.MODEL_MAP[args.model]().to(device)
    optimizer = optim.AdamW(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    df_train = create_df(args.path_train)
    df_val = create_df(args.path_val)
    df_test = create_df(args.path_test)

    train_ds = ds.DatasetSpectrogram(df_train, args.path_audio, audio2=audio2)

    val_ds = ds.DatasetSpectrogram(df_val, args.path_audio, audio2=audio2)

    test_ds = ds.DatasetSpectrogram(df_test, args.path_audio, audio2=audio2)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, shuffle=True)

    validation_loss, train_loss = train(model, train_loader, val_loader, optimizer, 
                                 criterion, device, epochs=args.epochs)
    acc = test(model, test_loader, criterion, device)
    ploting(args.model, args.spectrogram, validation_loss, train_loss, acc)
    torch.save(model.state_dict(), 'save/{}_{}_{:.2f}'.format(args.model, args.spectrogram, acc))