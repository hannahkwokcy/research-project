import sys
import os
import random
import glob
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression

print("RUNNING FROM:", sys.executable)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" 

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

start_time_raw = datetime.now()
start_time_str = start_time_raw.strftime("%Y-%m-%d %H:%M:%S")

cnn_model_name = "DenseNet121"
patience_val = 5
lr_p1 = 0.01
lr_p2 = 1e-06
class_weights = {0: 1.0, 1: 1.0}
epochs_p1_limit = 30
epochs_p2_limit = 50
batch_size = 64
n_splits = 5
class_mapping = ['Normal', 'Cataract']

train_dir = '/home/hlckwok2/research/dataset_0412/Train'
test_dir = '/home/hlckwok2/research/dataset_0412/Test'

def create_dataframe(directory):
    filepaths = []
    labels = []
    for cls in class_mapping:
        cls_dir = os.path.join(directory, cls)
        files = glob.glob(os.path.join(cls_dir, '*.*'))
        filepaths.extend(files)
        labels.extend([cls] * len(files))
    return pd.DataFrame({'filename': filepaths, 'class': labels})

train_df = create_dataframe(train_dir)
test_df = create_dataframe(test_dir)

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df, x_col='filename', y_col='class',
    target_size=(224, 224), batch_size=batch_size,
    class_mode='binary', classes=class_mapping, shuffle=False
)

def build_model():
    input_shape = (224, 224, 3)
    inputs = Input(shape=input_shape)
    base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    return Model(inputs, outputs), base_model

def gradcam_plus_plus(img_array, model, last_conv_layer_name='relu'):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None: return np.zeros((conv_outputs.shape[1], conv_outputs.shape[2]))

    conv_outputs, grads = conv_outputs[0], grads[0]
    grads_squared, grads_cubed = tf.square(grads), tf.square(grads) * grads
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

def save_gradcam_plots(pdf, indices, title_prefix, images, labels, preds, model):
    for i in indices:
        img = images[i]
        vis_img = (img - img.min()) / (img.max() - img.min())
        vis_img = vis_img[..., ::-1] 
        img_tensor = np.expand_dims(img, axis=0).astype(np.float32)

        heatmap = gradcam_plus_plus(img_tensor, model)
        heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(vis_img)
        im = ax.imshow(heatmap, cmap='jet', alpha=0.4, interpolation='bilinear')
        ax.set_title(f"{title_prefix}\nTrue: {labels[i]}, Pred: {preds[i]}")
        plt.colorbar(im, ax=ax)
        pdf.savefig(dpi=300)
        plt.close()

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

ensemble_test_probs = np.zeros(test_generator.samples)
metrics_history = {'loss': [], 'accuracy': [], 'precision': [], 'sensitivity': [], 'auc': [],
                   'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_sensitivity': [], 'val_auc': []}
best_auc = 0.0
best_model_path = f"best_fold_{cnn_model_name}.h5"
phase_1_lengths = []

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fold_specificities = []
fold_sensitivities = []
fold_accuracies = []

all_val_probs = []
all_val_labels = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df['filename'], train_df['class'])):
    print(f"\n{'='*40}\nStarting Fold {fold + 1}/{n_splits}\n{'='*40}")
    
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df.iloc[train_idx], x_col='filename', y_col='class',
        target_size=(224, 224), batch_size=batch_size, class_mode='binary', classes=class_mapping, shuffle=True
    )
    validation_generator = datagen.flow_from_dataframe(
        dataframe=train_df.iloc[val_idx], x_col='filename', y_col='class',
        target_size=(224, 224), batch_size=batch_size, class_mode='binary', classes=class_mapping, shuffle=False
    )

    model, base_model = build_model()

    model.compile(optimizer=Adam(learning_rate=lr_p1), loss='binary_crossentropy',
                  metrics=[BinaryAccuracy('accuracy'), Precision(name='precision'), Recall(name='sensitivity'), AUC(name='auc')])
    
    h1 = model.fit(train_generator, epochs=epochs_p1_limit, validation_data=validation_generator,
                   class_weight=class_weights, workers=8, use_multiprocessing=False,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=patience_val, restore_best_weights=True)])
    phase_1_lengths.append(len(h1.history['loss']))

    base_model.trainable = True
    
    for layer in base_model.layers[:-100]: 
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=lr_p2), loss='binary_crossentropy',
                  metrics=[BinaryAccuracy('accuracy'), Precision(name='precision'), Recall(name='sensitivity'), AUC(name='auc')])

    h2 = model.fit(train_generator, epochs=epochs_p2_limit, validation_data=validation_generator,
                   class_weight=class_weights, workers=8, use_multiprocessing=False,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=patience_val, restore_best_weights=True)])

    for metric in metrics_history.keys():
        combined_metric = h1.history[metric] + h2.history[metric]
        metrics_history[metric].append(combined_metric)

    validation_generator_unshuffled = datagen.flow_from_dataframe(
        dataframe=train_df.iloc[val_idx], x_col='filename', y_col='class',
        target_size=(224, 224), batch_size=batch_size, class_mode='binary', classes=class_mapping, shuffle=False
    )
    fold_val_probs = model.predict(validation_generator_unshuffled).flatten()
    all_val_probs.extend(fold_val_probs)
    all_val_labels.extend(validation_generator_unshuffled.classes)

    test_generator.reset()
    fold_probs = model.predict(test_generator).flatten()
    ensemble_test_probs += fold_probs / n_splits
    
    fpr_fold, tpr_fold, thresholds_fold = roc_curve(test_generator.classes, fold_probs)
    roc_auc_fold = auc(fpr_fold, tpr_fold)
    interp_tpr = np.interp(mean_fpr, fpr_fold, tpr_fold)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(roc_auc_fold)

    youden_j_fold = tpr_fold - fpr_fold
    opt_thresh_fold = thresholds_fold[np.argmax(youden_j_fold)]
    
    fold_preds = (fold_probs >= opt_thresh_fold).astype(int)
    tn_f, fp_f, fn_f, tp_f = confusion_matrix(test_generator.classes, fold_preds).ravel()
    
    fold_specificities.append(tn_f / (tn_f + fp_f))
    fold_sensitivities.append(tp_f / (tp_f + fn_f))
    fold_accuracies.append((tp_f + tn_f) / len(test_generator.classes))

    fold_auc = auc(*roc_curve(test_generator.classes, fold_probs)[:2])
    if fold_auc > best_auc:
        best_auc = fold_auc
        model.save(best_model_path)
        print(f"New best fold (AUC: {best_auc:.4f}), saving model.")

print("\npost-training calibration")
y_true = test_generator.classes

uncalibrated_probs = np.array(all_val_probs).reshape(-1, 1)
true_labels = np.array(all_val_labels)

calibrator = LogisticRegression(solver='lbfgs', penalty='none')
calibrator.fit(uncalibrated_probs, true_labels)

calibrated_ensemble_probs = calibrator.predict_proba(ensemble_test_probs.reshape(-1, 1))[:, 1]

fpr, tpr, thresholds = roc_curve(y_true, calibrated_ensemble_probs)
ensemble_auc = auc(fpr, tpr)

opt_threshold = 0.50

ensemble_preds_binary = (calibrated_ensemble_probs >= opt_threshold).astype(int)
opt_tn, opt_fp, opt_fn, opt_tp = confusion_matrix(y_true, ensemble_preds_binary).ravel()
opt_specificity = opt_tn / (opt_tn + opt_fp)
opt_sensitivity = opt_tp / (opt_tp + opt_fn)
opt_accuracy = (opt_tp + opt_tn) / len(y_true)
opt_precision = opt_tp / (opt_tp + opt_fp) if (opt_tp + opt_fp) > 0 else 0

report_str = classification_report(y_true, ensemble_preds_binary, target_names=class_mapping)

print(f"\nMetrics at 0.50 Threshold:")
print(report_str)
print(f"Calibrated Ensemble AUC: {roc_auc_score(y_true, calibrated_ensemble_probs):.4f}")

np.save(f"{cnn_model_name}_calibrated_probs.npy", calibrated_ensemble_probs)
np.save(f"{cnn_model_name}_true_labels.npy", y_true)

print("\nGenerating PDF Report")
with PdfPages(f'{cnn_model_name}_CV5_Full_Report_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pdf') as pdf:
    plt.rc('font', size=12)

    info_page = plt.figure(figsize=(11.69, 8.27))
    txt = f'''5-Fold Cross Validation Evaluation Report (Ensemble + Calibration)
Started: {start_time_str}
Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Model Architecture: {cnn_model_name}
Phase 1 LR: {lr_p1} | Phase 2 LR: {lr_p2}

Aggregate Performance across {n_splits} Folds (Individual Optimized Thresholds):
Average Accuracy:    {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}
Average Sensitivity: {np.mean(fold_sensitivities):.4f} ± {np.std(fold_sensitivities):.4f}
Average Specificity: {np.mean(fold_specificities):.4f} ± {np.std(fold_specificities):.4f}
Average AUC:         {np.mean(aucs):.4f} ± {np.std(aucs):.4f}

CALIBRATED ENSEMBLE TEST METRICS:
Forced Decision Threshold: {opt_threshold:.4f}
Accuracy                 : {opt_accuracy:.4f}
Precision                : {opt_precision:.4f}
Sensitivity (Recall)     : {opt_sensitivity:.4f}
Specificity              : {opt_specificity:.4f}
Ensemble AUC             : {ensemble_auc:.4f}

Classification Report (at threshold {opt_threshold:.4f}):
{report_str}
'''
    info_page.text(0.05, 0.98, txt, size=14, ha='left', va='top', family='monospace')
    pdf.savefig()
    plt.close()

    metrics_to_plot = ['loss', 'accuracy', 'precision', 'sensitivity', 'auc']
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        max_len = max([len(h) for h in metrics_history[metric]])
        
        padded_train = np.array([h + [h[-1]]*(max_len - len(h)) for h in metrics_history[metric]])
        padded_val = np.array([h + [h[-1]]*(max_len - len(h)) for h in metrics_history[f'val_{metric}']])
        
        mean_train = np.mean(padded_train, axis=0)
        mean_val = np.mean(padded_val, axis=0)
        
        plt.plot(mean_train, label=f'Mean Train {metric.capitalize()}', color='blue', lw=2)
        plt.plot(mean_val, label=f'Mean Val {metric.capitalize()}', color='orange', linestyle='--', lw=2)
        
        for i in range(n_splits):
            plt.plot(padded_train[i], color='blue', alpha=0.15)
            plt.plot(padded_val[i], color='orange', alpha=0.15, linestyle='--')

        avg_phase1_len = int(np.mean(phase_1_lengths))
        plt.axvline(x=avg_phase1_len - 1, color='red', linestyle=':', label='Avg Fine-Tuning Start')
        
        plt.title(f'Cross-Validated {metric.capitalize()} Over Epochs')
        plt.xlabel('Total Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True, alpha=0.3)
        pdf.savefig()
        plt.close()

    plt.figure(figsize=(11.69, 8.27))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    plt.plot(mean_fpr, mean_tpr, color='darkorange',
             label=r'Mean Fold ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc), lw=2)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='orange', alpha=0.2,
                     label=r'$\pm$ 1 std. dev. (Folds)')

    plt.plot(fpr, tpr, color='red', lw=2, linestyle=':', 
             label=f'Ensemble ROC (AUC = {ensemble_auc:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity (False Positive Rate)', fontweight='bold')
    plt.ylabel('Sensitivity (True Positive Rate)', fontweight='bold')
    plt.title(f'Cross-Validated ROC Curve - {cnn_model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    pdf.savefig()
    plt.close()

    print("\nGenerating Grad-CAM images")
    best_model = tf.keras.models.load_model(best_model_path)
    
    test_generator.reset()
    images, labels = next(test_generator) 
    
    preds_batch_probs = best_model.predict(images).flatten()
    preds_batch = (preds_batch_probs >= 0.5).astype(int)
    
    correct_idx = np.where(preds_batch == labels)[0]
    incorrect_idx = np.where(preds_batch != labels)[0]
    to_visualize = [
        ('Correctly Classified (Best Fold)', correct_idx[:10]),
        ('Incorrectly Classified (Best Fold)', incorrect_idx[:10])
    ]
    
    for title, indices in to_visualize:
        save_gradcam_plots(pdf, indices, title, images, labels, preds_batch, best_model)

    plt.figure(figsize=(11.69, 8.27))
    probs_normal = calibrated_ensemble_probs[y_true == 0]
    probs_cataract = calibrated_ensemble_probs[y_true == 1]
    
    plt.hist(probs_normal, bins=50, alpha=0.6, color='blue', label='True Normal')
    plt.hist(probs_cataract, bins=50, alpha=0.6, color='red', label='True Cataract')
    plt.axvline(x=opt_threshold, color='black', linestyle='--', label=f'Standard Threshold ({opt_threshold:.2f})')
    
    plt.title(f'{cnn_model_name} - Calibrated Ensemble Predicted Probabilities')
    plt.xlabel('Predicted Probability of Cataract')
    plt.ylabel('Number of Images')
    plt.legend()
    plt.grid(alpha=0.3)
    pdf.savefig()
    plt.close()
print('Summary PDF file created')