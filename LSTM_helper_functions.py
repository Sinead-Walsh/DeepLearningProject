import numpy as np
import torch
from distutils.version import LooseVersion as Version


def LSTM_train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        hidden = model.init_hidden()
        hidden = (hidden[0].to(device), hidden[1].to(device))

        outputs, hidden = model(inputs, hidden)

        _, predicted = torch.max(outputs, 1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_predictions / total_predictions

    return epoch_loss, epoch_accuracy


def LSTM_evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            hidden = model.init_hidden()
            hidden = (hidden[0].to(device), hidden[1].to(device))

            outputs, hidden = model(inputs, hidden)

            _, predicted = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            running_loss += loss.item()
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_predictions / total_predictions

    return epoch_loss, epoch_accuracy


def LSTM_test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            hidden = model.init_hidden()
            hidden = (hidden[0].to(device), hidden[1].to(device))

            outputs, hidden = model(inputs, hidden)

            _, predicted = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            running_loss += loss.item()
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    test_loss = running_loss / len(dataloader)
    test_accuracy = correct_predictions / total_predictions

    return test_loss, test_accuracy



