import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'image')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 7

def train_disease_model():
    print("=" * 60)
    print("STEP 1: Training Fast & Accurate Disease Classification Model")
    print("=" * 60)

    # 80/20 train/validation split with on-the-fly data augmentation
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.20,
        fill_mode='nearest'
    )

    train_gen = datagen.flow_from_directory(
        IMAGE_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_gen = datagen.flow_from_directory(
        IMAGE_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    class_indices = train_gen.class_indices
    print(f"Detected Classes: {class_indices}")
    with open(os.path.join(BASE_DIR, 'class_indices.json'), 'w') as f:
        json.dump(class_indices, f, indent=2)

    # Transfer Learning with MobileNetV2 pre-trained on ImageNet
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False  # Freeze base feature extractor for fast training

    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(len(class_indices), activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='Cocoa_Disease_MobileNetV2')

    model.compile(
        optimizer=Adam(learning_rate=0.0008),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    ]

    print(f"Starting training on {train_gen.samples} images, validating on {val_gen.samples} images...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\nFinal Disease Model Validation Accuracy: {val_acc * 100:.2f}% | Loss: {val_loss:.4f}")

    # Save in both .keras and .h5 formats
    h5_path = os.path.join(BASE_DIR, 'cocoa_disease_model.h5')
    keras_path = os.path.join(BASE_DIR, 'cocoa_disease_model.keras')

    model.save(keras_path)
    model.save(h5_path)
    print(f"Disease model successfully saved to:\n - {keras_path}\n - {h5_path}")

    return model, val_acc


def train_detector_model():
    print("\n" + "=" * 60)
    print("STEP 2: Training Fast Cocoa vs Non-Cocoa Detector Model")
    print("=" * 60)

    train_dir = os.path.join(DATASET_DIR, 'train')
    val_dir = os.path.join(DATASET_DIR, 'validation')

    # Explicit class order: index 0 -> non_cocoa, index 1 -> cocoa
    detector_classes = ['non_cocoa', 'cocoa']

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=detector_classes,
        class_mode='binary',
        shuffle=True,
        seed=42
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=detector_classes,
        class_mode='binary',
        shuffle=False,
        seed=42
    )

    print(f"Detector classes: {train_gen.class_indices}")

    num_non_cocoa = len(os.listdir(os.path.join(train_dir, 'non_cocoa')))
    num_cocoa = len(os.listdir(os.path.join(train_dir, 'cocoa')))
    total = num_non_cocoa + num_cocoa
    class_weights = {
        0: (total / (2.0 * num_non_cocoa)),
        1: (total / (2.0 * num_cocoa))
    }
    print(f"Class Weights: non_cocoa (0): {class_weights[0]:.2f}, cocoa (1): {class_weights[1]:.2f}")

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False

    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    detector = Model(inputs=inputs, outputs=outputs, name='Cocoa_Detector_MobileNetV2')
    detector.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)
    ]

    print("Training Cocoa Detector...")
    detector.fit(
        train_gen,
        validation_data=val_gen,
        epochs=4,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    val_loss, val_acc = detector.evaluate(val_gen, verbose=0)
    print(f"\nFinal Detector Validation Accuracy: {val_acc * 100:.2f}%")

    h5_path = os.path.join(BASE_DIR, 'cocoa_2.h5')
    keras_path = os.path.join(BASE_DIR, 'cocoa_2.keras')
    detector.save(keras_path)
    detector.save(h5_path)
    print(f"Detector model successfully saved to:\n - {keras_path}\n - {h5_path}")

    return detector, val_acc


if __name__ == '__main__':
    print("TensorFlow Version:", tf.__version__)
    train_disease_model()
    train_detector_model()
    print("\nAll models trained and exported successfully!")
