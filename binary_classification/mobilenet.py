import sys
print("RUNNING FROM:", sys.executable)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datetime import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from matplotlib.backends.backend_pdf import PdfPages
import cv2
import random

# 1. Record Start Time
start_time_raw = datetime.now()
start_time_str = start_time_raw.strftime("%Y-%m-%d %H:%M:%S")

# 3. Extract metadata for PDF
cnn_model_name = "MobileNetV2"
patience_val = 5
lr_p1 = 0.01
lr_p2 = 1e-6
weight = {0: 1.0, 1: 1.0}
# 4. Record target epochs
epochs_p1_limit = 30
epochs_p2_limit = 50

# ==========================
# Data Generators
# ==========================
SEED = 42  # You can pick any number, 42 is the standard meme

# 1. Set Python Hash Seed (must happen before TF is initialized)
os.environ['PYTHONHASHSEED'] = str(SEED)

# 2. Set Python's built-in pseudo-random generator
random.seed(SEED)

# 3. Set Numpy's random generator
np.random.seed(SEED)

# 4. Set TensorFlow's random generator
tf.random.set_seed(SEED)

batch_size = 64
class_mapping = ['Normal', 'Cataract']

train_datagen = ImageDataGenerator(
    validation_split=0.2,
    preprocessing_function=preprocess_input
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    '/home/hlckwok2/research/dataset_0412/Train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    classes=class_mapping,
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    '/home/hlckwok2/research/dataset_0412/Train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    classes=class_mapping,
    subset='validation',
    shuffle=True
)
test_generator = test_datagen.flow_from_directory(
    '/home/hlckwok2/research/dataset_0412/Test',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    classes=class_mapping,
    shuffle=False
)

print("Training Classes:", train_generator.class_indices)
print("Validation Classes:", validation_generator.class_indices)
print("Test Classes:", test_generator.class_indices)

# ==========================
# Model (Functional API)
# ==========================
input_shape = (224, 224, 3)
inputs = Input(shape=input_shape)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=lr_p1),
    loss='binary_crossentropy',
    metrics=[BinaryAccuracy('accuracy'),
             Precision(name='precision'),
             Recall(name='sensitivity'),
             AUC(name='auc')]
)

early_stopping = EarlyStopping(monitor='val_loss', patience=patience_val, restore_best_weights=True)
model.summary()
class_weights = weight

print("Phase 1: Training the Head...")
history_p1 = model.fit(
    train_generator,
    epochs=epochs_p1_limit,
    validation_data=validation_generator,
    class_weight = class_weights,
    workers=16,
    use_multiprocessing = True,  
    callbacks=[EarlyStopping(monitor='val_loss', patience=patience_val, restore_best_weights=True)]
)
actual_epochs_p1 = len(history_p1.history['loss'])

print("Phase 2: Unfreezing Top Layers...")
base_model.trainable = True
# Freeze layers up to the last block
for layer in base_model.layers[:140]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=lr_p2), # Lower LR for fine-tuning
    loss='binary_crossentropy',
    metrics=[BinaryAccuracy('accuracy'), Precision(name='precision'), 
             Recall(name='sensitivity'), AUC(name='auc')]
)

history_p2 = model.fit(
    train_generator,
    epochs=epochs_p2_limit,
    validation_data=validation_generator,
    class_weight = class_weights,
    workers=16,
    use_multiprocessing = True,  
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)
actual_epochs_p2 = len(history_p2.history['loss'])

# ==========================
# Grad-CAM++ function
# ==========================
def gradcam_plus_plus(img_array, model, last_conv_layer_name):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return np.zeros((conv_outputs.shape[1], conv_outputs.shape[2]))

    conv_outputs = conv_outputs[0]
    grads = grads[0]

    grads_squared = tf.square(grads)
    grads_cubed = grads_squared * grads

    global_sum = tf.reduce_sum(conv_outputs, axis=(0, 1))
    alpha_num = grads_squared
    alpha_denom = 2 * grads_squared + grads_cubed * global_sum
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))

    alphas = alpha_num / alpha_denom
    weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(0, 1))
    heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap) + 1e-10
    return heatmap.numpy()
# ==========================
# Save Grad-CAM++ plots
# ==========================
def save_gradcam_plots(pdf, indices, title_prefix, images, labels, preds, model):
    for i in indices:
        img = images[i]
        vis_img = (img - img.min()) / (img.max() - img.min())
        vis_img = vis_img[..., ::-1]
        img_tensor = np.expand_dims(img, axis=0).astype(np.float32)

        heatmap = gradcam_plus_plus(img_tensor, model, 'out_relu')
        heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(vis_img)
        im = ax.imshow(heatmap, cmap='jet', alpha=0.4, interpolation='bilinear')
        ax.set_title(f"{title_prefix}\nTrue: {labels[i]}, Pred: {preds[i]}")
        plt.colorbar(im, ax=ax)
        pdf.savefig(dpi=300)
        plt.close()
# ==========================
# Evaluate on test set
# ==========================
test_generator.reset()
images, labels = next(test_generator)
preds_probs = model.predict(images)
preds = (preds_probs > 0.5).astype(int).flatten()

correct_idx = np.where(preds == labels)[0]
incorrect_idx = np.where(preds != labels)[0]
to_visualize = [
    ('Correctly Classified', correct_idx[:10]),
    ('Incorrectly Classified', incorrect_idx[:10])
]

prediction_probs = model.predict(test_generator)
prediction = (prediction_probs > 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(test_generator.classes, prediction).ravel()
specificity = tn / (tn + fp)

results = model.evaluate(test_generator)
report_str = classification_report(test_generator.classes, prediction,
                                   target_names=test_generator.class_indices.keys())
print(f"Calculated Specificity: {specificity:.4f}")

#ROC curve
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)

fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

# ==========================
# Post-Training: Optimize Threshold using Youden's J
# ==========================
# 1. Calculate Youden's J statistic for all thresholds
youden_j = tpr - fpr

# 2. Find the index of the maximum J statistic
best_threshold_idx = np.argmax(youden_j)
optimal_threshold = thresholds[best_threshold_idx]

print(f"Default Threshold (0.5) metrics:")
print(f"Sensitivity: {tpr[np.argmin(np.abs(thresholds - 0.5))]:.4f}")
print(f"Specificity: {1 - fpr[np.argmin(np.abs(thresholds - 0.5))]:.4f}")

print(f"\nOptimal Threshold found: {optimal_threshold:.4f}")
print(f"Optimized Sensitivity: {tpr[best_threshold_idx]:.4f}")
print(f"Optimized Specificity: {1 - fpr[best_threshold_idx]:.4f}")

# 3. Apply the new optimal threshold to your predictions
optimized_prediction = (prediction_probs >= optimal_threshold).astype(int).flatten()

# Recalculate your final metrics using the optimized predictions
opt_tn, opt_fp, opt_fn, opt_tp = confusion_matrix(y_true, optimized_prediction).ravel()
opt_specificity = opt_tn / (opt_tn + opt_fp)
opt_sensitivity = opt_tp / (opt_tp + opt_fn)

print(f"Final Balanced Specificity: {opt_specificity:.4f}")
print(f"Final Balanced Sensitivity: {opt_sensitivity:.4f}")


# ==========================
# Save PDF report
# ==========================
def plot_combined_history(pdf, h1, h2, metric_name):
    # Combine the lists from both phases
    phase1_len = len(h1.history[metric_name])
    train_key = metric_name
    val_key = f'val_{metric_name}'
    
    # 3. Start Plotting
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    
    # Combine the lists from both phases
    train_metrics = h1.history[train_key] + h2.history[train_key]
    plt.plot(train_metrics, label=f'Train {metric_name}', color='blue')
    
    # Check if validation exists before plotting it
    if val_key in h1.history and val_key in h2.history:
        val_metrics = h1.history[val_key] + h2.history[val_key]
        plt.plot(val_metrics, label=f'Val {metric_name}', color='orange', linestyle='--')
    
    # 4. NOW use phase1_len (It's defined now!)
    plt.axvline(x=phase1_len - 1, color='red', linestyle=':', label='Fine-Tuning Starts')
    
    plt.title(f'{metric_name.capitalize()} Over Epochs')
    plt.xlabel('Total Epochs')
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    pdf.savefig()
    plt.close()


with PdfPages(f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pdf') as pdf:
    plt.rc('font', size=12, family='monospace')

    # Info page
    info_page = plt.figure(figsize=(11.69, 8.27))
    txt = f'''Model Evaluation Report
Started: {start_time_str}
Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Model Architecture: {cnn_model_name}
Early Stopping Patience: {patience_val}
Phase 1 Learning Rate: {lr_p1}
Phase 2 Learning Rate: {lr_p2}
Phase 1 Epochs:           {actual_epochs_p1} (Limit: {epochs_p1_limit})
Phase 2 Epochs:           {actual_epochs_p2} (Limit: {epochs_p2_limit})

{'Test Loss':<25}: {results[0]:.4f}
{'Binary Accuracy':<25}: {results[1]:.4f}
{'Precision':<25}: {results[2]:.4f}
{'Sensitivity (Recall)':<25}: {results[3]:.4f}
{'AUC':<25}: {results[4]:.4f}
Specificity: {specificity:.4f}

Optimal Threshold found: {optimal_threshold:.4f}
Optimised sensitivity: {opt_sensitivity:.4f}
Optimised specificity:{opt_specificity:.4f}

Classification Report:
{report_str}
'''
    info_page.text(0.05, 0.98, txt, size=15, ha='left', va='top')
    pdf.savefig()
    plt.close()

    # Training curves
    metrics_to_plot = ['accuracy', 'loss', 'sensitivity', 'auc', 'precision']
    for m in metrics_to_plot:
        plot_combined_history(pdf, history_p1, history_p2, m)

    # 3. Plotting
    plt.figure(figsize=(11.69, 8.27))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    pdf.savefig()
    plt.close()

    # ==========================
    # Generate Prediction Histogram
    # ==========================
    plt.figure(figsize=(11.69, 8.27))

    # Separate the probabilities based on their true class
    probs_normal = prediction_probs[y_true == 0]
    probs_cataract = prediction_probs[y_true == 1]

    # Plot overlapping histograms
    plt.hist(probs_normal, bins=50, alpha=0.6, color='blue', label='True Normal')
    plt.hist(probs_cataract, bins=50, alpha=0.6, color='red', label='True Cataract')

    # Add the optimal threshold line
    plt.axvline(x=optimal_threshold, color='black', linestyle='--', 
                label=f'Optimal Threshold ({optimal_threshold:.4f})')
    # Add standard 0.5 threshold line for comparison
    plt.axvline(x=0.5, color='gray', linestyle=':', label='Standard Threshold (0.5)')

    plt.title(f'{cnn_model_name} - Distribution of Predicted Probabilities')
    plt.xlabel('Predicted Probability of Cataract')
    plt.ylabel('Number of Images')
    plt.legend()
    plt.grid(alpha=0.3)
    pdf.savefig()
    plt.close()

    # Grad-CAM++ visualization
    for title, indices in to_visualize:
        save_gradcam_plots(pdf, indices, title, images, labels, preds, model)

print('Summary PDF file created')
np.savez(f'roc_data_{cnn_model_name}.npz', fpr=fpr, tpr=tpr, roc_auc=roc_auc)
# Combine the lists from Phase 1 and Phase 2
full_train_loss = history_p1.history['loss'] + history_p2.history['loss']
full_val_loss = history_p1.history['val_loss'] + history_p2.history['val_loss']

# Save to a dedicated file for this architecture
np.savez(f'loss_data_{cnn_model_name}.npz', 
         train_loss=full_train_loss, 
         val_loss=full_val_loss)