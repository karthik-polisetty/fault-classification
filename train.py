import tensorflow as tf
import matplotlib.pyplot as plt

def train_model(model, train_dataset, val_dataset, epochs=50, model_path="best_model.keras"):
    """
    Trains the CNN model with Learning Rate Scheduling and Early Stopping.
    Saves the model in `.keras` format.

    Args:
        model (tf.keras.Model): The compiled model.
        train_dataset (tf.data.Dataset): Training dataset.
        val_dataset (tf.data.Dataset): Validation dataset.
        epochs (int): Number of epochs.
        model_path (str): Filepath where the trained model will be saved.

    Returns:
        history (tf.keras.callbacks.History): Training history object.
    """

    # Learning Rate Scheduling
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    # Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping, lr_scheduler]
    )

    # Save the trained model as a single file (.keras or .h5)
    model.save(model_path)
    print(f"✅ Model saved at: {model_path}")

    return history


def evaluate_model(model, test_dataset):
    """
    Evaluates the trained model on the test dataset.

    Args:
        model (tf.keras.Model): Trained CNN model.
        test_dataset (tf.data.Dataset): Test dataset.

    Returns:
        Tuple (test_loss, test_accuracy): Model evaluation metrics.
    """
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return test_loss, test_accuracy

def plot_training_history(history):
    """
    Plots the training history (Accuracy & Loss) over epochs.

    Args:
        history (tf.keras.callbacks.History): Training history object.
    """
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')

    plt.show()