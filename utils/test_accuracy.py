import numpy as np
import torch

def test(outputs, y_test_original):
    """
    Evaluate the accuracy on the test set.
    y_test has 4 times as many labels as the original test data, y_test_original
    This is because our data augmentation converts one test time sample to 4. In effect, 
    y_test = [y_test_original y_test_original y_test_original y_test_original ]
    We compute the actual prediction by taking a majority vote
    """
    
    _, predictions = outputs.max(1)
    
    # Take the majority prediction for each sample
    predictions = predictions.reshape((4,-1))
    original_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
    
    # Now, do the comparison to y_test_original
    correct = (original_predictions == y_test_original).astype(int).sum()
    total = 443
    
    accuracy = 100*(correct/total)
    return accuracy


def compute_test_outputs(network, testloader, y_test):
    
    total_output = np.zeros_like(y_test)
    
    # Batch size is one, so every batch contains exactly one example.
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs = inputs.float()
        labels = labels.float()

        batch_output = network(inputs)
        total_output[i] = batch_output[0]
    
    return torch.from_numpy(total_output)
    