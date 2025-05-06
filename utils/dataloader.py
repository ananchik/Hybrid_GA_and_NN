from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import BATCH_SIZE


def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(
        datasets.MNIST('./data',
                       train=True,
                       download=False,
                       transform=transform),
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=True
    )

    test_loader = DataLoader(
        datasets.MNIST('./data',
                       train=False,
                       transform=transform),
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=False
    )

    return train_loader, test_loader
