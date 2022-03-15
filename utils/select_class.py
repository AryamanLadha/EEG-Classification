import numpy as np

def evaluate_on_classes(testloader):
    """
    Given a test set, see if we classify on one class better than the other one.
    This can help us see if our model is imbalanced, and biased towards any classes.
    There are 4 classes total.
    
    Inputs
    - testloader: the DataLoader for the test set
    
    Outputs
    - scores: an array of (loss, accuracy), where scores[i] are the loss and accuracy on the i'th example.
    """
    
    
    
    
    
    