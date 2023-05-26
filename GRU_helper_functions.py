import numpy as np
import torch
from distutils.version import LooseVersion as Version


def GRU_train(model, train_data, criterion, optimizer, device):
    model.train()
    model.to(device)

    train_loss = 0.0
    train_acc = 0.0
    total_samples = 0

    for inputs, labels in train_data:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        hidden = model.init_hidden()
        outputs, hidden = model(inputs, hidden)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_acc += torch.sum(preds == labels).item()
        total_samples += len(inputs)

    train_loss /= total_samples
    train_acc /= total_samples

    return train_loss, train_acc


def GRU_evaluate(model, data, criterion, device):
    model.eval()
    model.to(device)

    eval_loss = 0.0
    eval_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data:
            inputs = inputs.to(device)
            labels = labels.to(device)

            hidden = model.init_hidden()
            outputs, hidden = model(inputs, hidden)

            loss = criterion(outputs, labels)

            eval_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            eval_acc += torch.sum(preds == labels).item()
            total_samples += len(inputs)

    eval_loss /= total_samples
    eval_acc /= total_samples

    return eval_loss, eval_acc


def GRU_test(model, data, criterion, device):
    model.eval()
    model.to(device)

    test_loss = 0.0
    test_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data:
            inputs = inputs.to(device)
            labels = labels.to(device)

            hidden = model.init_hidden()
            outputs, hidden = model(inputs, hidden)

            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            test_acc += torch.sum(preds == labels).item()
            total_samples += len(inputs)

    test_loss /= total_samples
    test_acc /= total_samples

    return test_loss, test_acc



