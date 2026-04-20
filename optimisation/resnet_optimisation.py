import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import numpy as np
import os

# Headless plotting for HPC
matplotlib.use('Agg')

# ================================
# Parameters
# ================================
BATCH_SIZE = 16
TRAIN_DATA_PATH = '/home/hlckwok2/research/complete_dataset/Test'

# Phase 1: Feature Extraction
P1_MIN_LR = 1e-5
P1_MAX_LR = 1e-1
P1_EPOCHS = 5

# Phase 2: Fine-Tuning
P2_MIN_LR = 1e-7
P2_MAX_LR = 1e-3
P2_EPOCHS = 10

# ================================
# Improved LR Finder Callback
# ================================
class LRFinder(tf.keras.callbacks.Callback):
    def __init__(self, min_lr, max_lr, num_epochs, name="ResNet50"):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_epochs = num_epochs
        self.name = name
        self.lrs = []
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        # Corrected math: reaching max_lr at the final epoch
        lr = self.min_lr * (self.max_lr / self.min_lr) ** ((epoch + 1) / self.num_epochs)
        self.lrs.append(lr)
        self.losses.append(logs['loss'])
    
    def plot_lr_loss(self, filename):
        plt.figure(figsize=(12, 7))
        plt.plot(self.lrs, self.losses, linewidth=2, color='#1f77b4')
        
        # --- LOG SCALE: Crucial for seeing the "elbow" ---
        plt.xscale('log')
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        
        plt.grid(True, which="major", axis="both", linestyle='-', alpha=0.4)
        plt.grid(True, which="minor", axis="x", linestyle=':', alpha=0.2)

        plt.xlabel('Learning Rate (Log Scale)', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'LR Finder: {self.name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
       
        
    def get_best_lr(self):
        # Calculate gradients
        gradients = np.gradient(self.losses)
        
        # Skip the first 10% of the data to avoid "false starts" at the very beginning
        skip_idx = max(1, len(gradients) // 10)
        
        # Find the steepest descent in the remaining data
        best_idx = np.argmin(gradients[skip_idx:]) + skip_idx
        return self.lrs[best_idx]

# ================================
# Data Preparation
# ================================
train_datagen = ImageDataGenerator(
    validation_split=0.2,
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_PATH, target_size=(224, 224), batch_size=BATCH_SIZE, 
    class_mode='binary', subset='training'
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_PATH, target_size=(224, 224), batch_size=BATCH_SIZE, 
    class_mode='binary', subset='validation'
)

# ================================
# Phase 1: Feature Extraction Search
# ================================
print("--- Phase 1: Feature Extraction LR Search ---")
inputs = layers.Input(shape=(224, 224, 3))
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
base_model.trainable = False 

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

lr_finder_p1 = LRFinder(min_lr=P1_MIN_LR, max_lr=P1_MAX_LR, num_epochs=P1_EPOCHS, name="Phase 1 (Frozen)")
model.fit(train_generator, epochs=P1_EPOCHS, validation_data=val_generator, callbacks=[lr_finder_p1])

lr_finder_p1.plot_lr_loss("resnet_phase1_lr.png")
best_lr_p1 = lr_finder_p1.get_best_lr()
print(f"Optimal LR for Phase 1: {best_lr_p1:.6f}")

# ================================
# Phase 2: Fine-Tuning Search
# ================================
print("\n--- Phase 2: Fine-Tuning LR Search ---")
base_model.trainable = True
# Freeze early layers (standard ResNet practice)
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile with the base LR from phase 1
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr_p1), 
              loss='binary_crossentropy', metrics=['accuracy'])

lr_finder_p2 = LRFinder(min_lr=P2_MIN_LR, max_lr=P2_MAX_LR, num_epochs=P2_EPOCHS, name="Phase 2 (Fine-Tuning)")
model.fit(train_generator, epochs=P2_EPOCHS, validation_data=val_generator, callbacks=[lr_finder_p2])

lr_finder_p2.plot_lr_loss("resnet_phase2_lr.png")
best_lr_p2 = lr_finder_p2.get_best_lr()
print(f"Optimal LR for Phase 2: {best_lr_p2:.6f}")