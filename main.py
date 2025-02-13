from pathlib import Path
from data_loader import load_data, preprocess_data, split_data, create_datasets
from model import build_cnn_model
from train import train_model, evaluate_model, plot_training_history

# Define the dataset directory
DATA_DIR = Path("data")  # Update this path

# Load and preprocess data
X, y = load_data(DATA_DIR)
X, y = preprocess_data(X, y)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Create TensorFlow datasets
train_dataset, val_dataset, test_dataset = create_datasets(X_train, X_val, X_test, y_train, y_val, y_test)

# Model setup
input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = len(set(y))
model = build_cnn_model(input_shape, num_classes)

# Train model
history = train_model(model, train_dataset, val_dataset, epochs=25)

# Evaluate model
evaluate_model(model, test_dataset)

# Plot training history
plot_training_history(history)
