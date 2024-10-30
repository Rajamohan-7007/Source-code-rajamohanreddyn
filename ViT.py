import random
import os
import glob
import time
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchvision
import torchinfo
import scikitplot
import torch.optim.lr_scheduler as lr_scheduler

from torch import nn
from torch.utils.data import (Dataset, DataLoader)

from torchvision import transforms
from torchinfo import summary

from PIL import Image
from tqdm.notebook import tqdm
from typing import Dict, List, Tuple
from scikitplot.metrics import plot_roc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_recall_fscore_support,
    accuracy_score, top_k_accuracy_score, f1_score,
    matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
)
print("-----Packages are successfully installed-----")


class VisionTransformerModel(nn.Module):
    def __init__(self, backbone_model, name='vision-transformer',
                 num_classes=CFG.NUM_CLASSES, device=CFG.DEVICE):
        super(VisionTransformerModel, self).__init__()

        self.backbone_model = backbone_model
        self.device = device
        self.num_classes = num_classes
        self.name = name

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1000, out_features=256, bias=True),
            nn.GELU(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=256, out_features=num_classes, bias=False)
        ).to(device)

    def forward(self, image):
        vit_output = self.backbone_model(image)
        return self.classifier(vit_output)

def get_vit_b32_model(
    device: torch.device=CFG.NUM_CLASSES) -> nn.Module:
    # Set the manual seeds
    torch.manual_seed(CFG.SEED)
    torch.cuda.manual_seed(CFG.SEED)

    # Get model weights
    model_weights = (
        torchvision
        .models
        .ViT_L_32_Weights
        .DEFAULT
    )

    # Get model and push to device
    model = (
        torchvision.models.vit_l_32(
            weights=model_weights
        )
    ).to(device)

    # Freeze Model Parameters
    for param in model.parameters():
        param.requires_grad = False

    return model

# Get ViT model
vit_backbone = get_vit_b32_model(CFG.DEVICE)

vit_params = {
    'backbone_model'    : vit_backbone,
    'name'              : 'ViT-L-B32',
    'device'            : CFG.DEVICE
}

# Generate Model
vit_model = VisionTransformerModel(**vit_params)

# If using GPU T4 x2 setup, use this:
if CFG.NUM_DEVICES > 1:
    vit_model = nn.DataParallel(vit_model)


# View model summary
summary(
    model=vit_model,
    input_size=(CFG.BATCH_SIZE, CFG.CHANNELS, CFG.WIDTH, CFG.HEIGHT),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"]
)


# Define Loss Function
vit_loss_fn = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

# Define Optimizer
vit_optimizer = torch.optim.AdamW(
    vit_model.parameters(),
    lr=CFG.LR
)

# Train the model
print('Training ViT Model')
print(f'Train on {len(train_new_df)} samples, validate on {len(val_df)} samples.')
print('----------------------------------')

vit_session_config = {
    'model'               : vit_model,
    'train_dataloader'    : train_loader,
    'eval_dataloader'     : val_loader,
    'optimizer'           : vit_optimizer,
    'loss_fn'             : vit_loss_fn,
    'epochs'              : CFG.EPOCHS,
    'device'              : CFG.DEVICE
}

vit_session_history = train(**vit_session_config)

# Generate test sample probabilities
vit_test_probs = predict(vit_model, test_loader, CFG.DEVICE)

# Generate test sample preditions
vit_test_preds = np.argmax(vit_test_probs, axis=1)


def plot_training_curves(history):

    loss = np.array(history['loss'])
    val_loss = np.array(history['eval_loss'])

    accuracy = np.array(history['accuracy'])
    val_accuracy = np.array(history['eval_accuaracy'])

    epochs = range(len(history['loss']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot loss
    ax1.plot(epochs, loss, label='training_loss', marker='o')
    ax1.plot(epochs, val_loss, label='eval_loss', marker='o')

    ax1.fill_between(epochs, loss, val_loss, where=(loss > val_loss), color='C2', alpha=0.3, interpolate=True)
    ax1.fill_between(epochs, loss, val_loss, where=(loss < val_loss), color='C3', alpha=0.3, interpolate=True)

    ax1.set_title('Loss (Lower Means Better)', fontsize=16)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.legend()

    # Plot accuracy
    ax2.plot(epochs, accuracy, label='training_accuracy', marker='o')
    ax2.plot(epochs, val_accuracy, label='eval_accuracy', marker='o')

    ax2.fill_between(epochs, accuracy, val_accuracy, where=(accuracy > val_accuracy), color='C0', alpha=0.3, interpolate=True)
    ax2.fill_between(epochs, accuracy, val_accuracy, where=(accuracy < val_accuracy), color='C1', alpha=0.3, interpolate=True)

    ax2.set_title('Accuracy (Higher Means Better)', fontsize=16)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.legend();

    sns.despine();

    return


# Convert ViT history dict to DataFrame
vit_session_history_df = pd.DataFrame(vit_session_history)
vit_session_history_df


# Plot ViT session training history
plot_training_curves(vit_session_history)


def plot_confusion_matrix(y_true, y_pred, classes='auto', figsize=(10, 10), text_size=12):
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Set plot size
    plt.figure(figsize=figsize)

    # Create confusion matrix heatmap
    disp = sns.heatmap(
        cm, annot=True, cmap='Greens',
        annot_kws={"size": text_size}, fmt='g',
        linewidths=0.5, linecolor='black', clip_on=False,
        xticklabels=classes, yticklabels=classes)

    # Set title and axis labels
    disp.set_title('Confusion Matrix', fontsize=24)
    disp.set_xlabel('Predicted Label', fontsize=20)
    disp.set_ylabel('True Label', fontsize=20)
    plt.yticks(rotation=0)

    # Plot confusion matrix
    plt.show()

    return

test_labels = [*map(test_ds.class_to_idx.get, test_ds.labels)]


plot_confusion_matrix(
    test_labels,
    vit_test_preds,
    figsize=(16, 14),
    classes=test_ds.classes)

plot_roc(
    test_labels,
    vit_test_probs,
    figsize=(16, 10), title_fontsize='large'
);

print(
    classification_report(
        test_labels,
        vit_test_preds,
        target_names=test_ds.classes
))

# Generate EfficieNet model performance scores
vit_model_performance = generate_performance_scores(
    test_labels,
    vit_test_preds,
    vit_test_probs
)


# Record metrics with DataFrame
performance_df = pd.DataFrame({
    'efficientnet_v2_large': efficientnet_model_performance,
    'vit_l_b32': vit_model_performance,
}).T

# View Performance DataFrame
performance_df
