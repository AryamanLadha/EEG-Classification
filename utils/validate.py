import torch
import numpy as np

def validate(network, valloader, criterion):
    """
    Compute the validation loss and accuracy, using the network, valloader and loss criterion.
    Return validation loss and accuracy.
    For now, we compute this on the entire validation set, but later we might want to subsample to speed it up.
    """
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(valloader, 0):
        inputs, labels = data
        inputs = inputs.double() #has to be double to work with torch.nn criterion
        labels = labels.double()

        outputs = network(inputs)
        loss = criterion(outputs, labels)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(torch.argmax(labels, dim=1)).sum().item()
        running_loss += loss.item()

    val_loss=running_loss/len(valloader)
    val_accuracy=100.*correct/total
    
    return val_accuracy, val_loss
    

