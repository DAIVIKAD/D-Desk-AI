"""
Train or test the image classifier using the best transfer-learning CNN.

Usage:
    python app/ml/image_train.py
    python app/ml/image_train.py --test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.ml.image_model import (
    FEEDBACK_ROOT,
    IMAGE_SIZE,
    LABELS_PATH,
    MODEL_PATH,
    SUPPORTED_IMAGE_LABELS,
    TEST_ROOT,
    TRAIN_ROOT,
    ensure_image_directories,
)


MIN_IMAGES_PER_CLASS = 100
BATCH_SIZE = 16
EPOCHS = 5
VALIDATION_SPLIT = 0.2
BALANCE_MIN_PER_CLASS = 120
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _ensure_tf():
    try:
        import tensorflow as tf
        return tf
    except Exception:
        return None


def _collect_samples(*roots: Path) -> list[tuple[str, str]]:
    samples = []
    for root in roots:
        if not root.exists():
            continue
        for label_dir in root.iterdir():
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            if label not in SUPPORTED_IMAGE_LABELS:
                continue
            for file_path in label_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES:
                    samples.append((str(file_path), label))
    return samples


def _count_samples(root: Path) -> dict[str, int]:
    counts = {label: 0 for label in SUPPORTED_IMAGE_LABELS}
    if not root.exists():
        return counts

    for label in SUPPORTED_IMAGE_LABELS:
        label_dir = root / label
        if not label_dir.exists():
            continue
        counts[label] = sum(
            1
            for file_path in label_dir.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES
        )
    return counts


def _print_dataset_readiness(train_counts: dict[str, int]) -> None:
    print("Dataset readiness (train/)")
    print("-" * 56)
    for label in SUPPORTED_IMAGE_LABELS:
        count = train_counts.get(label, 0)
        status = "OK" if count >= MIN_IMAGES_PER_CLASS else "NEEDS_DATA"
        print(f"{label}: {count} images [{status}]")

    missing_or_small = [
        label for label, count in train_counts.items() if count < MIN_IMAGES_PER_CLASS
    ]
    if missing_or_small:
        print("-" * 56)
        print(
            "Warning: the following labels are below the recommended minimum "
            f"of {MIN_IMAGES_PER_CLASS} images each:"
        )
        print(", ".join(missing_or_small))
        print(
            "Training will continue with the available labeled images so the "
            "existing pipeline remains backward compatible."
        )


def _split_samples(
    samples: list[tuple[str, str]],
    validation_split: float = VALIDATION_SPLIT,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    grouped_samples: dict[str, list[tuple[str, str]]] = {}
    rng = random.Random(42)

    for sample in samples:
        grouped_samples.setdefault(sample[1], []).append(sample)

    train_samples: list[tuple[str, str]] = []
    val_samples: list[tuple[str, str]] = []

    for label in sorted(grouped_samples):
        label_samples = list(grouped_samples[label])
        rng.shuffle(label_samples)

        if len(label_samples) <= 1:
            train_samples.extend(label_samples)
            continue

        val_count = max(1, int(len(label_samples) * validation_split))
        if len(label_samples) - val_count < 1:
            val_count = len(label_samples) - 1

        val_samples.extend(label_samples[:val_count])
        train_samples.extend(label_samples[val_count:])

    if not val_samples and train_samples:
        val_samples.append(train_samples.pop())

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def _balance_training_samples(samples: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Oversample small classes for training only.

    This keeps validation honest while preventing rare classes such as
    cable_issue and mouse_issue from being drowned out by larger folders.
    """
    grouped_samples: dict[str, list[tuple[str, str]]] = {}
    rng = random.Random(42)
    for sample in samples:
        grouped_samples.setdefault(sample[1], []).append(sample)

    if not grouped_samples:
        return samples

    target_count = max(BALANCE_MIN_PER_CLASS, max(len(items) for items in grouped_samples.values()))
    balanced: list[tuple[str, str]] = []
    for label in sorted(grouped_samples):
        items = grouped_samples[label]
        balanced.extend(items)
        if len(items) < target_count:
            balanced.extend(rng.choice(items) for _ in range(target_count - len(items)))

    rng.shuffle(balanced)
    return balanced


def _build_dataset(
    tf,
    samples: list[tuple[str, str]],
    label_to_index: dict[str, int],
    batch_size: int,
    training: bool,
):
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    paths = [sample[0] for sample in samples]
    labels = [label_to_index[sample[1]] for sample in samples]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = preprocess_input(tf.cast(image, tf.float32))
        return image, label

    dataset = dataset.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(len(samples), seed=42, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def _build_transfer_model(tf, base_model_class, model_name: str, num_classes: int):
    model_slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in model_name)

    base_model = base_model_class(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name=f"{model_slug}_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_image_model(selected_model: str | None = None, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    ensure_image_directories()
    tf = _ensure_tf()
    if tf is None:
        print("TensorFlow is not installed. Install tensorflow-macos/tensorflow-metal or tensorflow first.")
        return 1

    from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50

    tf.keras.utils.set_random_seed(42)

    train_counts = _count_samples(TRAIN_ROOT)
    _print_dataset_readiness(train_counts)

    samples = _collect_samples(TRAIN_ROOT, FEEDBACK_ROOT)
    if len(samples) < 2:
        print("Not enough training images found. Add labeled images under app/ml/image_data/train/<label>/ first.")
        return 1

    labels = sorted({label for _, label in samples})
    if len(labels) < 2:
        print("At least two image classes are required to train a classifier.")
        return 1

    label_to_index = {label: index for index, label in enumerate(labels)}
    train_samples, val_samples = _split_samples(samples)
    balanced_train_samples = _balance_training_samples(train_samples)

    train_ds = _build_dataset(tf, balanced_train_samples, label_to_index, batch_size=batch_size, training=True)
    val_ds = _build_dataset(tf, val_samples, label_to_index, batch_size=batch_size, training=False)

    model_builders = {
        "MobileNetV2": MobileNetV2,
        "EfficientNetB0": EfficientNetB0,
        "ResNet50": ResNet50,
    }
    if selected_model:
        if selected_model not in model_builders:
            print(f"Unknown model '{selected_model}'. Choose one of: {', '.join(model_builders)}")
            return 1
        model_builders = {selected_model: model_builders[selected_model]}
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
        ),
    ]

    print("Training image classifiers...")
    print(f"Training samples: {len(train_samples)}")
    print(f"Balanced training samples: {len(balanced_train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Active labels: {', '.join(labels)}")

    results = {}
    for model_name, base_model_class in model_builders.items():
        print("-" * 56)
        print(f"Training {model_name}...")
        model = _build_transfer_model(tf, base_model_class, model_name, len(labels))
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        _, accuracy = model.evaluate(val_ds, verbose=0)
        results[model_name] = (model, float(accuracy))
        print(f"{model_name} validation accuracy: {accuracy:.2%}")

    print("-" * 56)
    print("📊 Model Comparison:")
    for model_name, (_, accuracy) in results.items():
        print(f"{model_name}: {accuracy:.2%}")

    best_model_name = max(results, key=lambda name: results[name][1])
    best_model, best_acc = results[best_model_name]
    print(f"🏆 Best Model: {best_model_name} ({best_acc:.2%})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_model.save(MODEL_PATH)
    LABELS_PATH.write_text(json.dumps(labels, indent=2))

    print(f"Saved best image model to {MODEL_PATH}")
    print(f"Saved labels to {LABELS_PATH}")
    return 0


def test_image_model():
    ensure_image_directories()
    tf = _ensure_tf()
    if tf is None:
        print("TensorFlow is not installed. Install tensorflow-macos/tensorflow-metal or tensorflow first.")
        return 1
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        print("No trained image model found. Run without --test first.")
        return 1

    labels = json.loads(LABELS_PATH.read_text())
    label_to_index = {label: index for index, label in enumerate(labels)}
    samples = _collect_samples(TEST_ROOT)
    if not samples:
        print("No test images found under app/ml/image_data/test/.")
        return 1

    dataset = _build_dataset(tf, samples, label_to_index, batch_size=1, training=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    predictions = model.predict(dataset, verbose=0)

    print(f"Image test predictions ({model.name})")
    print("-" * 56)
    for (file_path, true_label), probs in zip(samples, predictions):
        pred_index = int(probs.argmax())
        pred_label = labels[pred_index]
        confidence = float(probs[pred_index])
        print(f"{Path(file_path).name}: predicted={pred_label} confidence={confidence:.2%} actual={true_label}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the best transfer-learning image classifier.")
    parser.add_argument("--test", action="store_true", help="Run predictions for images under app/ml/image_data/test/.")
    parser.add_argument("--model", choices=["MobileNetV2", "EfficientNetB0", "ResNet50"], help="Train one architecture instead of comparing all three.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    args = parser.parse_args()

    raise SystemExit(
        test_image_model()
        if args.test
        else train_image_model(selected_model=args.model, epochs=args.epochs, batch_size=args.batch_size)
    )
