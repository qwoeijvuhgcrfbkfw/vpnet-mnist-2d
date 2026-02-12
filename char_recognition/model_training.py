import copy
from typing import Tuple, Callable

import torch


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_train(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader[Tuple[torch.Tensor, ...]],
    n_epoch: int,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor | list[torch.Tensor], torch.Tensor], torch.Tensor],
    quiet: bool = False,
    callback: Callable[[torch.nn.Module, int], ...] = None
) -> None:
    model.train()
    if not quiet:
        print(f"Starting training with {count_parameters(model)} learnable parameters")

    n_digits = len(str(n_epoch))

    for epoch in range(n_epoch):
        total_loss = 0
        total_accuracy = 0
        total_number = 0

        for data in data_loader:
            x, labels = data

            optimizer.zero_grad()

            outputs = model(x)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            classes = labels.argmax(dim=-1)
            y = outputs[0] if isinstance(outputs, tuple) else outputs
            y_classes = y.argmax(dim=-1)
            total_accuracy += (classes == y_classes).sum().item()
            total_number += labels.size(0)
        total_accuracy /= total_number / 100

        if not quiet:
            print(f"Epoch: {epoch + 1:0{n_digits}d} / {n_epoch}, accuracy: {total_accuracy:.2f}%, loss: {total_loss:.4f}")

        if callback is not None:
            callback(model, epoch)

    if not quiet:
        print("Finished training.")

def model_test(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader[Tuple[torch.Tensor, ...]],
    criterion: Callable[[torch.Tensor | list[torch.Tensor], torch.Tensor], torch.Tensor],
    quiet: bool = False
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_accuracy = 0
        total_number = 0
        for data in data_loader:
            x, labels = data
            outputs = model(x)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            classes = labels.argmax(dim=-1)
            y = outputs[0] if isinstance(outputs, tuple) else outputs
            y_classes = y.argmax(dim=-1)
            total_accuracy += (classes == y_classes).sum().item()
            total_number += labels.size(0)
        total_accuracy /= total_number / 100

        if not quiet:
            print(f"Test accuracy: {total_accuracy:.2f}%, loss: {total_loss:.4f}")

        return total_accuracy, total_loss

def model_train_best_test_accuracy(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader[Tuple[torch.Tensor, ...]],
    valid_loader: torch.utils.data.DataLoader[Tuple[torch.Tensor, ...]],
    test_loader: torch.utils.data.DataLoader[Tuple[torch.Tensor, ...]],
    n_epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor | list[torch.Tensor], torch.Tensor], torch.Tensor],
    quiet: bool = False
) -> float:
    max_acc: float = 0
    best_model: torch.nn.Module = None

    def callback(curr_model: torch.nn.Module, epoch: int):
        nonlocal max_acc, best_model

        valid_acc, _ = model_test(curr_model, valid_loader, criterion, quiet)

        if valid_acc > max_acc:
            max_acc = valid_acc
            best_model = copy.deepcopy(curr_model)

    model_train(model, train_loader, n_epochs, optimizer, criterion, quiet, callback)

    if not quiet:
        print(f"Best valid accuracy: {max_acc:.2f}%")

    test_acc, _ = model_test(best_model, test_loader, criterion, quiet)

    if not quiet:
        print(f"Test accuracy: {test_acc:.2f}%")

    return test_acc