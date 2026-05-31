"""MobileNetV2 transfer learning — species image classifier."""

from __future__ import annotations

import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "processed"))
LOG_DIR = Path(os.getenv("LOG_DIR", "logs/mobilenet"))
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 30


def make_generators() -> tuple[
    tf.keras.preprocessing.image.DirectoryIterator,
    tf.keras.preprocessing.image.DirectoryIterator,
    tf.keras.preprocessing.image.DirectoryIterator,
]:
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    eval_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    kwargs = dict(target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")
    return (
        train_gen.flow_from_directory(PROCESSED_DIR / "train", **kwargs),
        eval_gen.flow_from_directory(PROCESSED_DIR / "val", **kwargs),
        eval_gen.flow_from_directory(PROCESSED_DIR / "test", shuffle=False, **kwargs),
    )


def build_model(num_classes: int) -> Model:
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )
    return model


def train() -> tf.keras.callbacks.History:
    train_gen, val_gen, _ = make_generators()
    num_classes = len(train_gen.class_indices)

    Path("checkpoints").mkdir(exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            "checkpoints/mobilenet_best.keras",
            save_best_only=True,
            monitor="val_loss",
        ),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=str(LOG_DIR)),
    ]

    model = build_model(num_classes)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    model.save("checkpoints/mobilenet_final.keras")
    return history


if __name__ == "__main__":
    train()
