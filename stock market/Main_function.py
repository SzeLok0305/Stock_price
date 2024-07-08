#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:37:09 2024

@author: arnold
"""

from Generate_plot import *

InterestedTimeSeries = 'Close'

HK0388_History = Load_Data(FlieLocation="Data/HK0388/past_data/price.csv",ExtractLabel=InterestedTimeSeries)
HK0388_Truth = Load_Data(FlieLocation="Data/HK0388/new_data/price.csv",ExtractLabel=InterestedTimeSeries)

#%%
number_of_walk = 10000

Plot_Starting_date = datetime.date(2015,8,4)
Prediction_Ending_date = datetime.date(2024,5,20)
Sample_start_date=datetime.date(2017,8,4)
Sample_end_date=datetime.date(2023,8,4)
Plot_history_ending_date = Sample_end_date
plot_true_stock_price(HK0388_Truth,Sample_end_date)
Plot_stock_price(Time_series_history=HK0388_History,Time_serise_name="HK0388",Plot_from=Plot_Starting_date,Plot_to=Plot_history_ending_date,T_Prediction_sample_start=Sample_start_date,T_Prediction_sample_end=Sample_end_date,T_Prediction_End=Prediction_Ending_date,N_walk=number_of_walk,reweighting_function="none")
#%%
NVDA_History = Load_Data(FlieLocation="Data/NVDA/past_data/price.csv",ExtractLabel=InterestedTimeSeries)
NVDA_Truth = Load_Data(FlieLocation="Data/NVDA/new_data/price.csv",ExtractLabel=InterestedTimeSeries)

#%%
number_of_walk = 26330 #  include a 5.5 sigma event = 1 in 26330254
Plot_Starting_date = datetime.date(2017,8,4)
Sample_start_date=datetime.date(2020,8,4)
Sample_end_date=datetime.date(2024,6,10)
Plot_history_ending_date = Sample_end_date
Prediction_Ending_date = datetime.date(2024,8,1)
plot_true_stock_price(NVDA_Truth,Sample_end_date)
Plot_stock_price(Time_series_history=NVDA_History,Time_serise_name="NVDA",Plot_from=Plot_Starting_date,Plot_to=Plot_history_ending_date,T_Prediction_sample_start=Sample_start_date,T_Prediction_sample_end=Sample_end_date,T_Prediction_End=Prediction_Ending_date,N_walk=number_of_walk,reweighting_function="sigmod")

#%%
number_of_walk = 100
Test_price = lambda x: 0.0004*x**2 + 0.0002*x**1
time_test = np.linspace(1, 2343,2343)
Test_price_mono_increase = Test_price(time_test)
Test_history = [Test_price_mono_increase,NVDA_History[1][4044:]]

plot_true_stock_price(NVDA_Truth,Sample_end_date)
Plot_stock_price(Time_series_history=Test_history,Time_serise_name="Test",Plot_from=Plot_Starting_date,Plot_to=Plot_history_ending_date,T_Prediction_sample_start=Sample_start_date,T_Prediction_sample_end=Sample_end_date,T_Prediction_End=Prediction_Ending_date,N_walk=number_of_walk)



