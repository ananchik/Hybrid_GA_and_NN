import torch
import torch.nn as nn
import numpy as np
import random
import time
from config import (
    SEED,
    DEVICE,
    GENERATIONS,
    TRAINING_STEPS,
    LEARNING_RATE,
)
from utils.dataloader import load_mnist
import torch.optim as optim
from networks.models import MNISTModel
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from utils.logger import ResultLogger


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        if batch == TRAINING_STEPS: break
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:

    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        'training_time': [],
        'total_evaluation_time': [],
        'total_time': [],
    }
    custom_logger = ResultLogger(experiment_name="baseline_10steps")

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        epoch_start = time.time()

        train_start = time.time()
        train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
        train_time = time.time() - train_start

        eval_start = time.time()
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device)
        eval_time = time.time() - eval_start

        epoch_time = time.time() - epoch_start
        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.2%} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.2%}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["training_time"].append(train_time)
        results["total_evaluation_time"].append(eval_time)
        results["total_time"].append(epoch_time)

        custom_logger.log_generation(
            metrics={
                'phase': 'full_cycle_baseline',
                'generation': epoch,

                'best_gd_loss': test_loss,
                'best_gd_accuracy': test_acc,

                'training_time': train_time,
                'total_evaluation_time': eval_time,
                'total_time': epoch_time,
            },
        )

    # Return the filled results at the end of the epochs
    return results


def main():
    # Initialisation
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    loss_fn = nn.CrossEntropyLoss()
    model = MNISTModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_dataloader, test_dataloader = load_mnist()

    # Start training with help from engine.py
    results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=GENERATIONS,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()