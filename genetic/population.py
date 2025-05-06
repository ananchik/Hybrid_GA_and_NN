import numpy as np
import torch
from tqdm import tqdm
from networks.models import MNISTModel
from config import DEVICE
from typing import Tuple


class Population:
    def __init__(self, size=100,):
        self.size = size
        self.models = [MNISTModel().to(DEVICE) for _ in range(size)]
        self.fitness = np.zeros((size, 2))

    @staticmethod
    def _evaluate_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device,
    ) -> Tuple[float, float]:
        # Put model in eval mode
        model.eval()

        # Setup test loss and test accuracy values
        test_loss, test_acc = 0, 0

        # Turn on inference context manager
        with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)

                # 1. Forward pass
                test_pred_logits = model(X)

                # 2. Calculate and accumulate loss
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()

                # Calculate and accumulate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

        # Adjust metrics to get average loss and accuracy per batch
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)

        return test_loss, test_acc

    def evaluate_generation(
        self,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device,
    ):
        for i, model in enumerate(tqdm(self.models, desc="Evaluate")):
            self.fitness[i] = self._evaluate_step(model, data_loader, loss_fn, device)

    def get_parameters(self,):
        return [np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()]) for model in self.models]

    def set_parameters(self, parameters):
        for model, params in zip(self.models, parameters):
            idx = 0
            with torch.no_grad():
                for p in model.parameters():
                    shape = p.shape  # torch.Size([128, 784])
                    size = p.numel()  # size 100352
                    p.copy_(
                        torch.tensor(params[idx:idx+size]).reshape(shape)
                    )  # Back to weight format
                    idx += size  # idx = 100352
