import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

def build_cnn_model(input_shape, num_classes):
    """
    Builds an improved 1D CNN model with Batch Normalization, Dropout, and L2 Regularization.

    Args:
        input_shape (tuple): Shape of input data (time_steps, channels).
        num_classes (int): Number of output classes.

    Returns:
        model (tf.keras.Model): Compiled CNN model.
    """
    model = Sequential([
        # 1st Convolutional Layer
        Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01), input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),  # Dropout layer

        # 2nd Convolutional Layer
        Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # 3rd Convolutional Layer
        Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Flatten(),

        # Fully Connected Layer
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),

        Dense(num_classes, activation='softmax')  # Output Layer
    ])

    # Compile Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
