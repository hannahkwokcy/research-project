import numpy as np
import matplotlib.pyplot as plt


models = ['ResNet50', 'MobileNetV2', 'DenseNet121']
colors = ['blue', 'green', 'red']


plt.figure(figsize=(6, 4))
for model_name, color in zip(models, colors):
    try:
        data = np.load(f'roc_data_{model_name}.npz')
        plt.plot(data['fpr'], data['tpr'], color=color, lw=2, 
                    label=f'{model_name} (AUC = {data["roc_auc"]:.3f})')
    except FileNotFoundError:
        pass

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curve Model Comparison')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('ROC_curve.png')
plt.close()

# ---------------------------------------------------------
# PAGE 2: Validation Loss Overlay
# ---------------------------------------------------------
plt.figure(figsize=(6, 4))

phase2_starts = {
        'ResNet50': 18,
        'MobileNetV2': 23, 
        'DenseNet121': 27   
    }

for model_name, color in zip(models, colors):
    try:
        # Load the combined history data
        loss_data = np.load(f'loss_data_{model_name}.npz')
        val_loss = loss_data['val_loss']
        
        # Generate the X-axis (Epochs 1 through N)
        epochs = range(1, len(val_loss) + 1)
        
        # Plot the Validation Loss
        plt.plot(epochs, val_loss, color=color, lw=2, label=f'{model_name}')
        
        if model_name in phase2_starts:
            start_epoch = phase2_starts[model_name]
            y_val = val_loss[start_epoch - 1]
            plt.plot(start_epoch, y_val, marker='o', markersize=8, 
                        color=color, markeredgecolor='black', linestyle='None', zorder=5)

    except FileNotFoundError:
        print(f"Warning: Loss data for {model_name} not found. Skipping.")

plt.title('Validation Loss Over Epochs Comparison')
plt.xlabel('Total Epochs (Phase 1 + Phase 2)')
plt.ylabel('Validation Loss')
plt.legend(loc="upper right", fontsize=8)
plt.grid(True, alpha=0.3)

plt.savefig('validation_loss.png')
plt.close()
