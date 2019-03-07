# File: training_functions.py
# Author: Ronil Pancholia
# Date: 3/7/19
# Time: 2:46 AM

import sys
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_batch(model, criterion, x, y, opt=None):
    outputs = model(x)
    loss = criterion(outputs, y)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    preds = torch.argmax(outputs.data, 1)
    corrects = torch.sum(preds == y)
    return loss.item(), len(x), corrects


def fit(num_epochs, model, criterion, opt, train_dataloader, val_dataloader=None):
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):
        print("\nEpoch " + str(epoch + 1))

        running_loss = 0.0
        model.train()
        running_corrects = 0
        for image, label in tqdm(train_dataloader, file=sys.stdout):
            image, label = image.to(device), label.to(device)
            losses, nums, corrects = loss_batch(model, criterion, image, label, opt)
            running_loss += losses
            running_corrects += corrects

        train_loss.append(running_loss / len(train_dataloader.dataset))
        train_acc.append(running_corrects.item() / (len(train_dataloader.dataset)))
        print("Training loss:", train_loss[-1])
        print("Training accuracy: %.2f" % train_acc[-1])

        if val_dataloader == None:
            continue

        model.eval()
        running_corrects = 0
        with torch.no_grad():
            for image, label in tqdm(val_dataloader, file=sys.stdout):
                image, label = image.to(device), label.to(device)
                losses, nums, corrects = loss_batch(model, criterion, image, label)
                running_loss += losses
                running_corrects += corrects

        val_loss.append(running_loss / len(val_dataloader.dataset))
        val_acc.append(running_corrects.item() / (len(val_dataloader.dataset)))
        print("Validation loss:", val_loss[-1])
        print("Validation accuracy: %.2f" % val_acc[-1])

    return train_loss, train_acc, val_loss, val_acc