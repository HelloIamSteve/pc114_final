import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from model import LeNet

@torch.no_grad()
def test(model, test_loader, device):
    model.eval()
    correct = 0
    no_correct = 0
    total = 0

    for inputs, labels in tqdm(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model(inputs)
        correct += (output.argmax(dim=1) == labels).sum().item()
        no_correct += (output.argmax(dim=1) != labels).sum().item()
        total += labels.size(0)

    acc = correct / total * 100

    return acc

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('using device', device)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))])

    # dataset
    dataset_test = torchvision.datasets.MNIST(root=config.dataset_root_dir, train=False, download=True, transform=transform)

    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=512, shuffle=False)
    
    model = LeNet().to(device)
    model.load_state_dict(torch.load(f'./{model.name}_best.pt'))

    criterion = nn.CrossEntropyLoss()

    acc = test(model, loader_test, device)
    print(f'accuracy: {acc:.3f}%')