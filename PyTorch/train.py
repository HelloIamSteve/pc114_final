import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from model import LeNet

def train(model, train_loader, optimizer, criterion, device):
    model.train()

    loss_total = 0

    for inputs, labels in tqdm(train_loader, position=1, leave=False):
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model(inputs)
        loss = criterion(output, labels)
        loss_total += loss.item()
        
        loss.backward()
        optimizer.step()

    loss_avg = loss_total / len(train_loader)

    return loss_avg

@torch.no_grad()
def valid(model, val_loader, criterion, device):
    model.eval()

    loss_total = 0

    for inputs, labels in tqdm(val_loader, position=1, leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model(inputs)
        loss = criterion(output, labels)
        loss_total += loss.item()

    loss_avg = loss_total / len(val_loader)

    return loss_avg

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('using device', device)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))])

    # dataset
    train_val_ratio=(0.8, 0.2)
    dataset_train = torchvision.datasets.MNIST(root=config.dataset_root_dir, train=True, download=True, transform=transform)

    # split dataset
    train_len = int(train_val_ratio[0] * len(dataset_train))
    val_len = len(dataset_train) - train_len
    dataset_train, dataset_val = torch.utils.data.random_split(dataset_train, [train_len, val_len])

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False)

    model = LeNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

    # training loop
    loss_val_list = []
    loss_val_best = float('inf')

    for _ in tqdm(range(config.epoch_num), position=0, leave=True):
        train(model, loader_train, optimizer, criterion, device)

        loss_val = valid(model, loader_val, criterion, device)
        if loss_val < loss_val_best:
            torch.save(model.state_dict(), f'./{model.name}_best.pt')
            loss_val_best = loss_val

        loss_val_list.append(loss_val)
    
    plt.plot(loss_val_list)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss over Epochs')
    plt.show()
    print('finish')