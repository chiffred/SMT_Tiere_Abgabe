import cv2 as cv
import mediapipe as mp
import time
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os


class HumanimalClassifier(nn.Module):
    def __init__(self, in_feat=0, hiddenlayer=0,
                 num_classes=0):  # Start with three classes and adjust as your project evolves.
        super().__init__()
        self.layer1 = nn.Linear(in_feat, hiddenlayer)
        self.layer2 = nn.Linear(hiddenlayer, hiddenlayer//2)
        self.layer3 = nn.Linear(hiddenlayer//2, num_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)  # No need activation here as normally the loss function will take care of it
        return x