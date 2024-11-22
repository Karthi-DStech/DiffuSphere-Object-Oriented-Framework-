import random
import os

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Sets the seed for the experiment

    Parameters
    ----------
    seed: int
        The seed to use for the experiment
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    
def delete_files_in_directory(directory: str) -> None:
    """
    Deletes all files in a directory

    Parameters
    ----------
    directory: str
        The directory to delete the files from
    """
    # Iterate over all the files and subdirectories in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            # Check if it's a file
            if os.path.isfile(file_path):
                # Delete the file
                os.remove(file_path)
            # If it's a directory, recursively delete its content
            elif os.path.isdir(file_path):
                delete_files_in_directory(file_path)
                # After deleting all files in the subdirectory, remove the subdirectory itself
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {str(e)}")

def delete_directory(directory: str) -> None:
    """
    Deletes a directory

    Parameters
    ----------
    directory: str
        The directory to delete
    """
    try:
        # Remove the directory itself
        os.rmdir(directory)
        print(f"Directory '{directory}' and its contents deleted successfully.")
    except Exception as e:
        print(f"Failed to delete directory '{directory}'. Reason: {str(e)}")