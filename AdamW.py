import numpy as np
import random
import os
import glob
import time
import warnings

import pandas as pd
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
------------------------------------------------------------------
class AdamW:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, dw):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        self.t += 1
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * dw
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dw ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        # Compute bias-corrected second moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Weight decay
        w -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * w)
        
        return w

# Example usage
if __name__ == "__main__":
    # Example: weight and gradient
    weight = np.array([0.5, -0.5])
    gradient = np.array([0.1, -0.2])

    optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
    
    # Update weights
    new_weight = optimizer.update(weight, gradient)
    print("Updated Weights:", new_weight)
--------------------------------------------------------------------------------------
import tensorflow as tf

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),  # Example input shape
    tf.keras.layers.Dense(1)
])

# Compile the model using AdamW
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
              loss='mean_squared_error')

# Example training data
import numpy as np
x_train = np.random.rand(1000, 32)  # 1000 samples, 32 features
y_train = np.random.rand(1000, 1)    # 1000 target values

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

