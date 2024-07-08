#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:30:29 2024

@author: arnold
"""
import Load_data as LD
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
import MC_Sim

def Generated_plots_for_sims(Stocks,day,stock_label,stock_name,color):
    plt.plot(day,Stocks[0],label=stock_label,color=color,alpha=0.5)
    for i in range(len(Stocks)): 
        plt.plot(day,Stocks[i],color=color,alpha=2.5/len(Stocks))
    #plt.xlabel("Date")
    plt.ylabel("price")
    plt.title(f'{stock_name}')
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    return 

def Generated_plots(Stocks,days,stock_label,stock_name,colors):
    for i in range(len(Stocks)): 
        #print(stock_label[i])
        plt.plot(days[i],Stocks[i],label=stock_label[i],color=colors[i])
    plt.xlabel("Date")
    plt.ylabel("price")
    plt.title(f'{stock_name}')
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    return 

def Generate_error_plot(Walks, time_for_plot, colors, box_wide):
    error_distibution = []
    for j in range(len(Walks)):
        error_distibution.append(Walks[j][-1])
        
    error_distibution = np.array(error_distibution)
    x_pos = mdates.date2num(max(time_for_plot)) # Adjust the position of the error bar plot
    #y_pos =  np.mean(error_distibution) # Adjust the y-position of the error bar plot
    errorboxplot = plt.boxplot(error_distibution, positions=[x_pos], widths=box_wide, patch_artist=True, boxprops=dict(facecolor=colors, color='black', linewidth=2), medianprops=dict(color='black'),showfliers=True)
    for patch in errorboxplot['boxes']:
        patch.set_alpha(0.5)

    return error_distibution, errorboxplot

def get_box_plot_data(distribution):
    quartiles = np.quantile(distribution, [0.25,0.5,0.75])
    q1 = quartiles[0]
    mean = quartiles[1]  # Median (also known as the second quartile)
    q3 = quartiles[2]
    return mean, q1, q3

def plot_districtuion(histogram,title):
    plt.hist(histogram,density=False,bins="auto")
    plt.ylabel("frequency")
    plt.xlabel("final price")
    plt.title(title)
    plt.show()

def Plot_stock_price(Time_series_history,Time_serise_name,Plot_from,Plot_to,T_Prediction_sample_start,T_Prediction_sample_end,T_Prediction_End,N_walk,reweighting_function):
    X1 = Time_series_history[0]
    T = Time_series_history[1]
    #Get the history
    time_History, price_history = MC_Sim.Slice_time_series(X1,T,Plot_from,Plot_to)
    time_History = MC_Sim.Convert_str_into_date(time_History)
    time_prediction, X_prediction_ave, X_predictions = MC_Sim.Get_prediction(Time_series=X1,Time_day=T,N_walk=N_walk,T_Prediction_sample_start=T_Prediction_sample_start,T_Prediction_sample_end=T_Prediction_sample_end,T_Prediction_End=T_Prediction_End,reweighting_function="none")
    time_prediction, X_prediction_ave_weighted, X_predictions_weighted = MC_Sim.Get_prediction(Time_series=X1,Time_day=T,N_walk=N_walk,T_Prediction_sample_start=T_Prediction_sample_start,T_Prediction_sample_end=T_Prediction_sample_end,T_Prediction_End=T_Prediction_End,reweighting_function=reweighting_function)
    
    final_averaged_price = X_prediction_ave[-1]
    final_averaged_price_weighted = X_prediction_ave_weighted[-1]
    
    plot_stock_list = [price_history, X_prediction_ave, X_prediction_ave_weighted]
    plot_days_list = [time_History, time_prediction, time_prediction]
    plot_stock_label = ["history", "simple prediction", "weighted prediction"]
    Plot_stock_colors = ['blue', 'red','green']
    
    # Plot the distribution oif the final position of the walks as an error bar plot
    error_distibution, bp = Generate_error_plot(X_predictions,time_prediction, "red",50)
    error_distibution_weighted, bp_weighted= Generate_error_plot(X_predictions_weighted,time_prediction, "green",50)
    ylimit_plot = max(np.concatenate((X_prediction_ave,price_history)))
    plt.ylim(0,ylimit_plot*1.5)
    
    Generated_plots(plot_stock_list,plot_days_list,plot_stock_label,Time_serise_name,Plot_stock_colors)
    
    # Adjust the x-axis limits to include the boxplot
    plt.xlim(min(time_History), max(time_prediction) + datetime.timedelta(days=100))  # Adjust the padding as needed
    plt.legend()    
    plt.show()
    
    plot_districtuion(error_distibution,"unweighed")
    plot_districtuion(error_distibution_weighted,"activation: " + reweighting_function)
    #plt.hist(error_distibution)
    
    mean, q1, q3 = get_box_plot_data(error_distibution)
    mean_weighted, q1_weighted, q3_weighted = get_box_plot_data(error_distibution)
    print("Number of MC sample = ", N_walk, " activation function is", reweighting_function)
    print(f"Predicting {Time_serise_name} at", T_Prediction_End)
    
    print("------------------------------------------------------------------")
    print(f"final averaged central value of {Time_serise_name} is", round(final_averaged_price,1), "(mean at", round(mean,1))
    print("q1 - q3 range is", [round(q1,1),round(q3,1)])
    print("Max-min values are", [round(max(error_distibution),1),round(min(error_distibution),1)])
    print("------------------------------------------------------------------")
    print(f"final averaged central value of {Time_serise_name} (weighted) is", round(final_averaged_price_weighted,1), "(mean at", round(mean_weighted,1),")")
    print("q1 - q3 range is", [round(q1_weighted,1),round(q3_weighted,1)])
    print("Max-min values are", [round(max(error_distibution_weighted),1),round(min(error_distibution_weighted),1)])
    return

def plot_true_stock_price(True_stock,start_from,label_name,color):
    
    True_stock_price = True_stock[0]
    T_true = True_stock[1]
    
    start_index_true = MC_Sim.Find_day_index(T_true, start_from)
    Time_string_true = T_true[start_index_true:]
    Time_true = MC_Sim.Convert_str_into_date(Time_string_true)
    
    Stock_price = True_stock_price[start_index_true:]
    
    plt.plot(Time_true,Stock_price,label=label_name,color = color)
    
    #print("The price at ", Time_string_true[-1], "is ", Stock_price[-1])
    return