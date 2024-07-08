#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 19:20:03 2024

@author: arnold
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import matplotlib.pyplot as plt

import NN_training

#%%

MAV_model = NN_training.Load_model("NVDA","Moving_average_volatility_order/VO1_30_days")
