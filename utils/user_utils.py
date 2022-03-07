import numpy as np

def get_idxs(in_array, to_find):
    """Utility function for finding the positions of observations of one array in another an array.

    Args:
        in_array (array): Array in which to locate elements of to_find
        to_find (array): Array of elements to locate in in_array

    Returns:
        TYPE: Indices of all elements of to_find in in_array
    """
    targets = ([np.where(in_array == x) for x in to_find])
    return np.squeeze(np.concatenate(targets, axis=1))
