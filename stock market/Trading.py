#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 19:32:36 2024

@author: arnold
"""
import numpy as np
import MC_Sim
import Load_data
#%%

def Get_Moving_Gaussian_parameters(Stock_price,averaging_days):
    Grow_rates = []
    Volatilities = []
    for i in range(len(Stock_price)):
        Diff_Stock_price = MC_Sim.Precentage_Del(Stock_price)
        start_index = max(i - averaging_days + 1, 0)
        Subject_Stock_price_resticted = Diff_Stock_price[start_index:i+1]
        #Subject_Stock_price_resticted = Stock_price[start_index:i+1]
        
        expected_grow_rate = np.mean(Subject_Stock_price_resticted)
        volatility = np.var(Subject_Stock_price_resticted)
        
        Grow_rates.append(expected_grow_rate)
        Volatilities.append(volatility)
        
    Grow_rates=np.array(Grow_rates)
    Volatilities=np.array(Volatilities)
    return Grow_rates, Volatilities

def Get_Gaussian_parameters(Stock_price):
    #Diff_Stock_price = MC_Sim.Precentage_Del(Stock_price)
    Grow_rate = np.mean(Stock_price)
    Volatility = np.var(Stock_price)
    return Grow_rate, Volatility

def Binary_Kelly(probability_increase,possible_value_increase,possible_value_decrease):
    probability_decrease = 1-probability_increase
    betting_fraction = probability_increase/possible_value_decrease - probability_decrease/possible_value_increase
    return betting_fraction

def Non_Binary_Kelly(expected_grow_rate,volatility,remaining_rate_of_return):
    betting_fraction = (expected_grow_rate-remaining_rate_of_return)/volatility
    return betting_fraction

def Get_Non_Binary_Kelly_f(Stock_price,Full_time_data,on_which_day,averaging_days):
    Grow_rates, Volatilities = Get_Moving_Gaussian_parameters(Stock_price,averaging_days)
    index = Load_data.Find_day_index(Full_time_data,on_which_day)
    f = Non_Binary_Kelly(Grow_rates[index],Volatilities[index],0)
    return f

def Buying_share(Current_price,Cash,current_holding,fraction):
    share=current_holding
    availability = Cash*fraction
    buyable_share = availability//Current_price
    if buyable_share >= 1:
        share = buyable_share
        Cash = Cash - Current_price*buyable_share
    return share, Cash


def Buy_and_hold(Principle,Stock_open):
    Cash = Principle
    buyable_share = Cash//Stock_open[0]
    Cash = Cash - buyable_share*Stock_open[0]
        
    stock_rpice_record = Stock_open*buyable_share
    Cash_record = np.array(np.repeat(Cash, len(stock_rpice_record)))
    
    return buyable_share, stock_rpice_record, Cash_record


def Trade(Principle,Stock_open, NN_scorce, betting_fraction, Call_scorce, Short_scorce,opt_mode=False):
    Cash = Principle
    share = 0
    N_call = 0
    N_short = 0
    N_no_action = 0
    
    share_worth_record = []
    cash_worth_record = []
    if betting_fraction > 1: #I don't have enough money
        betting_fraction = 1
    
    #betting_strategy = "Non_Binary_Kelly"
    
    for jj in range(len(NN_scorce)):
        if NN_scorce[jj] >= Call_scorce:
            N_call += 1
            #availability = Cash*betting_fraction
            #buyable_share = availability//Stock_open[jj]
            #if buyable_share >= 1:
                #share = buyable_share
                #Cash = Cash - Stock_open[jj]*buyable_share
            share, Cash = Buying_share(Stock_open[jj],Cash,share,betting_fraction)
            
        elif NN_scorce[jj] < Short_scorce:
            N_short+= 1
            #print("detected close call, scorce at",NN_scorce[jj] )
            if share >= 1:
                Cash = Cash + Stock_open[jj]*share
                share = 0
        else:
            N_no_action += 1
        
        #print("At ",jj,"step cash and share are ", (Cash,share))
        #update worth
        share_worth_record.append(share*Stock_open[jj])
        cash_worth_record.append(Cash)
            
    
    share_worth_record = np.array(share_worth_record)
    cash_worth_record = np.array(cash_worth_record)
    
    
    worth_record = cash_worth_record + share_worth_record
    diff_worth = (worth_record[1:]-worth_record[:-1])/worth_record[:-1]
    worth_loss_function = np.mean(diff_worth)
    
    if opt_mode:
        #print("Current worth", worth)xs
        return worth_loss_function
    else:
        return share, Cash, share_worth_record, cash_worth_record, N_call, N_short, N_no_action
    
def optimize_parameters(Principle, Stock_open, NN_scorce,betting_fraction):
    best_worth = -np.inf
    best_call_scorce = 0
    best_short_scorce = 0
    
    delta_step = 0.01
    for short_scorce in np.arange(0, 1, delta_step):
        for call_scorce in np.arange(0, short_scorce+delta_step,  delta_step):
            worth = Trade(Principle, Stock_open, NN_scorce, betting_fraction, call_scorce, short_scorce,opt_mode=True)
            
            if worth > best_worth:
                best_worth = worth
                best_call_scorce = call_scorce
                best_short_scorce = short_scorce
                #print("Current best combination:",short_scorce,call_scorce)
    
    return best_call_scorce, best_short_scorce

def optimize_parameters_within_timeframe(Principle,Full_time_step, Stock_price, NN_scorce,betting_fraction,Time_start,Time_end):
    Time_step_Opt_1, Opt_Open_restricted = MC_Sim.Slice_time_series(Stock_price,Full_time_step,Time_start,Time_end)
    
    #Scorce is mis-matched my +1 day compared to the Open
    Time_step_Opt_2, Opt_scorce_restricted = MC_Sim.Slice_time_series(NN_scorce,Full_time_step[1:],Time_start,Time_end)

    Call_scorce_opt, Short_scorce_opt = optimize_parameters(Principle, Opt_Open_restricted, Opt_scorce_restricted, betting_fraction)
    
    return Call_scorce_opt, Short_scorce_opt

    
    
    
    
    
    
    
