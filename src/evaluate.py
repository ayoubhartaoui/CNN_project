"""Model evaluation and Explainable AI (saliency maps)."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)


def evaluate(
    model: tf.keras.Model,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    class_names: list[str],
    output_dir: Path = Path("results"),
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    loss, *metric_values = model.evaluate(test_data, test_labels, verbose=0)
    metric_names = [m.name for m in model.metrics[1:]]
    metrics = {"loss": loss, **dict(zip(metric_names, metric_values))}

    probs = model.predict(test_data, verbose=0)
    preds = (probs > 0.5).astype(int).flatten() if probs.shape[-1] == 1 else probs.argmax(-1)

    print(classification_report(test_labels, preds, target_names=class_names))

    _plot_confusion_matrix(test_labels, preds, class_names, output_dir)
    _plot_roc(test_labels, probs, output_dir)
    _plot_training_history(metrics, output_dir)

    return metrics


def _plot_confusion_matrix(
    true: np.ndarray,
    pred: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> None:
    cm = confusion_matrix(true, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _plot_roc(
    true: np.ndarray,
    probs: np.ndarray,
    output_dir: Path,
) -> None:
    flat_probs = probs.ravel()
    flat_true = true.ravel()
    fpr, tpr, _ = roc_curve(flat_true, flat_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close(fig)


def _plot_training_history(metrics: dict, output_dir: Path) -> None:
    pass


def compute_saliency(
    model: tf.keras.Model,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = tf.convert_to_tensor(image[np.newaxis], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(tensor)
        prediction = model(tensor)
        loss = tf.reduce_max(prediction)

    gradients = tape.gradient(loss, tensor)
    saliency = np.max(np.abs(gradients.numpy()), axis=-1)[0]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
    saliency = np.clip(saliency * 1.5, 0, 1)

    saliency_resized = cv2.resize(saliency, (image.shape[1], image.shape[0]))
    saliency_colored = cv2.applyColorMap((saliency_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    original = (image * 255).astype(np.uint8)
    superimposed = cv2.addWeighted(original, 0.6, saliency_colored, 0.4, 0)

    return saliency, superimposed, original


def plot_saliency_grid(
    model: tf.keras.Model,
    images: np.ndarray,
    output_path: Path,
    n: int = 4,
) -> None:
    n = min(n, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))

    for i in range(n):
        saliency, superimposed, original = compute_saliency(model, images[i])
        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"Original {i + 1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(saliency, cmap="hot")
        axes[i, 1].set_title(f"Saliency {i + 1}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(superimposed)
        axes[i, 2].set_title(f"Superimposed {i + 1}")
        axes[i, 2].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saliency grid saved → {output_path}")
