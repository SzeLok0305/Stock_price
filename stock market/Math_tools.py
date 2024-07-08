#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 02:16:30 2024

@author: arnold
"""
import numpy as np
import matplotlib.pyplot as plt
def Alpha_Beta(x, y,plot=False):
    # Fit a linear polynomial (degree 1) to the data(Linear regression)
    coefficients = np.polyfit(x, y, deg=1)
    beta = coefficients[0]
    beta = np.round(beta,decimals=3)
    
    alpha = coefficients[1]
    alpha = np.round(alpha,decimals=3)
    if plot:
        
        y_pred = np.polyval(coefficients, x)
    
        plt.scatter(x, y, label='Original Data')
        plt.plot(x, y_pred, color='red', label='Linear Regression')
        plt.xlabel('Strategy')
        plt.ylabel('Control')
        plt.legend()
        plt.show()
    return alpha,beta