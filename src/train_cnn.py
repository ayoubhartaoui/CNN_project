"""Custom CNN — binary classifier for Amphiprion ocellaris detection."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from dataset import load_split


PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "processed"))
LOG_DIR = Path(os.getenv("LOG_DIR", "logs/cnn"))
BATCH_SIZE = 32
EPOCHS = 30
L2_LAMBDA = 0.01


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3),
                      kernel_regularizer=l2(L2_LAMBDA)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(L2_LAMBDA)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", kernel_regularizer=l2(L2_LAMBDA)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu", kernel_regularizer=l2(L2_LAMBDA)),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
        ],
    )
    return model


def augment(
    data: np.ndarray,
    labels: np.ndarray,
    n_augments: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
    )
    aug_data, aug_labels = [], []
    for img, label in zip(data, labels):
        flow = gen.flow(img[np.newaxis], batch_size=1)
        for _ in range(n_augments):
            aug_data.append(next(flow)[0])
            aug_labels.append(label)
    return (
        np.concatenate([data, np.array(aug_data)]),
        np.concatenate([labels, np.array(aug_labels)]),
    )


def train() -> tf.keras.callbacks.History:
    train_data, train_labels = load_split(PROCESSED_DIR / "train")
    val_data, val_labels = load_split(PROCESSED_DIR / "val")

    train_data, train_labels = augment(train_data, train_labels)

    class_weights = dict(enumerate(
        compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    ))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        tf.keras.callbacks.TensorBoard(log_dir=str(LOG_DIR)),
        tf.keras.callbacks.ModelCheckpoint(
            "checkpoints/cnn_best.keras", save_best_only=True, monitor="val_loss"
        ),
    ]

    model = build_model()
    history = model.fit(
        train_data, train_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(val_data, val_labels),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    model.save("checkpoints/cnn_final.keras")
    return history


if __name__ == "__main__":
    train()
