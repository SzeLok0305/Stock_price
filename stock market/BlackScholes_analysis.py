#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:39:34 2024

@author: arnold
"""

from Load_data import *
from scipy.optimize import curve_fit
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

#Close, T_data = Load_Data(FlieLocation="Data/HK0388/past_data/price.csv",ExtractLabel='Close')

#Close_true, T_dat_true = Load_Data(FlieLocation="Data/HK0388/new_data/price.csv",ExtractLabel='Close')

InterestedTimeSeries = 'Close'

NVDA_History = Load_Data(FlieLocation="Data/NVDA/past_data/price.csv",ExtractLabel=InterestedTimeSeries)
NVDA_Truth = Load_Data(FlieLocation="Data/NVDA/new_data/price.csv",ExtractLabel=InterestedTimeSeries)

#%%

"""
   We use Euler-Maruyama method to solve the SDE

"""


def StockModel(S0, mu, sigma, dt, dW):
    return S0 + mu/S0 * dt + sigma * S0**(1/2) * dW


def fit_function(t, S0, mu, sigma):
    #Euler-Maruyama method
    S = np.zeros(t.size)
    S[0] = S0
    for i in range(1, t.size):
        dt = t[i] - t[i - 1]
        dW = np.random.normal(loc=0.0, scale=np.sqrt(dt))
        S[i] = StockModel(S[i - 1], mu, sigma, dt, dW)
    return S


def fit_data(t_data, S_data):
    optimal_params, _ = curve_fit(fit_function, t_data, S_data, p0=[S_data[0], 0.1, 0.2])
    return optimal_params


NVDA_price = NVDA_History[0][5000:]
Time_index = np.linspace(0,len(NVDA_price),len(NVDA_price))


# Fit the data over N-simulation
def simulationN(Stock, Time, N_simulation):
    
    S_predictions = []
    mu_fits = []
    sigma_fits = []
    for i in range(N_simulation):
        
        optimal_params = fit_data(Time, Stock)
        S0_fit, mu_fit, sigma_fit = optimal_params
        S_predicted = fit_function(Time_index,S0_fit, mu_fit, sigma_fit)
        mu_fits.append(mu_fit)
        sigma_fits.append(sigma_fit)
        S_predictions.append(S_predicted)
    
    #average over prediction
    S_predictions = np.array(S_predictions)
    S_prediction_ave = np.mean(S_predictions, axis=0)

    # Print the estimated parameters
    print("mean mu:", np.mean(sigma_fits))
    print("mean sigma:", np.mean(sigma_fits))

    # Plot the results
    plt.plot(Time_index, NVDA_price, label="True")
    plt.plot(Time_index, S_prediction_ave, label="average prediction")
    plt.xlabel("Time")
    plt.ylabel("S")
    
    for S in S_predictions:
        plt.plot(Time_index, S, alpha = 0.11,color = "g")
    plt.plot(Time_index, S_predictions[0], label="predictions", alpha = 0.11,color = "g")
    plt.legend()
    plt.show()
    
    return S_predictions, mu_fits, sigma_fits

A,B,C = simulationN(NVDA_price,Time_index,100)

