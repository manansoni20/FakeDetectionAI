# src/preprocess.py
import pandas as pd

def load_data(file_path=None):
    """
    Load data from CSV file if file_path is given,
    else return dummy data.
    """
    if file_path:
        # load CSV into DataFrame
        df = pd.read_csv(file_path)
        return df
    else:
        # fallback dummy data
        import numpy as np
        texts = ["Hello", "Fake news", "AI demo"]
        images = [np.zeros((64,64,3)), np.ones((64,64,3)), np.ones((32,32,3))]
        return texts, images


