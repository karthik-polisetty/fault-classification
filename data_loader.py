import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_data(data_dir: Path):
    """
    Load dataset from the specified directory.

    Args:
        data_dir (Path): Path to the dataset directory.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Loaded feature (X) and label (y) arrays.
    """
    x_path = data_dir / "X_t.npy"
    y_path = data_dir / "y_t.npy"

    X = np.load(x_path)
    y = np.load(y_path)

    print(f"Loaded Data - X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def preprocess_data(X, y):
    """
    Preprocess data by transposing feature dimensions.

    Args:
        X (np.ndarray): Feature array.
        y (np.ndarray): Label array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Transformed feature and label arrays.
    """
    X = np.transpose(X, (0, 2, 1))  # Reshape: (samples, channels, time) → (samples, time, channels)
    print(f"Transformed X shape: {X.shape}")

    y_series = pd.Series(y)
    print(f"Label Distribution:\n{y_series.value_counts()}")

    return X, y

def split_data(X, y, test_size=0.3, val_size=0.5, random_state=42):
    """
    Split dataset into train, validation, and test sets.

    Args:
        X (np.ndarray): Feature array.
        y (np.ndarray): Label array.
        test_size (float): Test set size fraction.
        val_size (float): Validation set size fraction.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple: Train, validation, and test splits of features and labels.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_datasets(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    """
    Convert NumPy arrays to TensorFlow datasets.

    Args:
        X_train, X_val, X_test (np.ndarray): Feature sets.
        y_train, y_val, y_test (np.ndarray): Label sets.
        batch_size (int): Batch size for training.

    Returns:
        Tuple: TensorFlow dataset objects for training, validation, and testing.
    """
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).shuffle(1000)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    return train_dataset, val_dataset, test_dataset
