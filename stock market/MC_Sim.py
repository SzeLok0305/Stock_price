# -*- coding: utf-8 -*-
"""
Created on Tue May 14 18:48:38 2024

@author: User
"""

import Load_data as LD
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
from scipy.optimize import fsolve

#Close, T_data = Load_Data(FlieLocation="Data/HK0388/past_data/price.csv",ExtractLabel='Close')

#Close_true, T_dat_true = Load_Data(FlieLocation="Data/HK0388/new_data/price.csv",ExtractLabel='Close')

#InterestedTimeSeries = 'Close'

#History = LD.Load_Data(FlieLocation="Data/HK0388/past_data/price.csv",ExtractLabel=InterestedTimeSeries)
#Truth = LD.Load_Data(FlieLocation="Data/HK0388/new_data/price.csv",ExtractLabel=InterestedTimeSeries)


#%% 

def Make_days_for_prediction(T_prediction_start,T_prediction_end):
    T_final = datetime.datetime.strptime(T_prediction_start, "%Y-%m-%d").date()
    Prediton_date_list = [T_final + datetime.timedelta(days=x) for x in range((T_prediction_end - T_final).days + 1)] 
    step = len(Prediton_date_list)
    return Prediton_date_list, step

def Convert_str_into_date(Time_frame):
    date_list =[]
    for Time_string in Time_frame:
        T_date = datetime.datetime.strptime(Time_string, "%Y-%m-%d").date()
        date_list.append(T_date)
    return date_list
    
def Derivative(Time_series,plot=False):
    X1 = Time_series
    D_X1 = []
    D_X1.append(np.float64(0)) # Set initial value
    for i in range(len(X1)-1):
        dx = X1[i+1]-X1[i]
        D_X1.append(dx)
    if plot:
        plt.hist(D_X1,bins="auto",density="True", alpha=1, label='True distribution')
        plt.xlabel("Price changes")
        plt.title(f'Plot of Delta {Time_series}')
        plt.show()
    
    return D_X1

def Precentage_Del(Time_series):
    X1 = Time_series
    D_X1 = []
    #D_X1.append(np.float64(0)) # Set initial value
    for i in range(len(X1)-1):
        dx_precentage = (X1[i+1]-X1[i])/X1[i]
        D_X1.append(dx_precentage)
    return D_X1

def Random_Walk(D_Time_series, step):
    # Generate random samples based on the histogram
    num_steps = step  # Define the number of steps in the random walk
    walk = np.random.choice(D_Time_series, size=num_steps, replace=True)
    
    return walk

def Weighted_random_walk(D_Time_series,probabilities,step):
    reweighted_walk = np.random.choice(D_Time_series, size=step, replace=True, p=probabilities)
    return reweighted_walk

def Precentage_Random_Walk(D_Time_series, step):
    # Generate random samples based on the histogram
    num_steps = step  # Define the number of steps in the random walk
    samples = np.random.choice(D_Time_series, size=num_steps, replace=True)
    
    # Perform the random walk
    walk = samples
    #check_nan_values(samples)

    # Plot the random walk trajectory
    #plt.plot(walk, label='Random Walk')
    #plt.show()
    return walk

"""

        Time-weighted bias walk:
            The basic idea is to re-weight the Delta_S histogram using proximity in time.
            The closer it is to present the higher the weight should be.
            

"""

def Signmod_Coefficient(Last_data_Importance, Lenght_of_sample):
    alpha = 1/Lenght_of_sample # For satablity reason, I neet to fix alpha to prevent overflow issue
    def equations(vars):
        A, B = vars     
        eq1 = A/(1 + np.exp(-alpha)) + B - 1
        eq2 = A/(1 + np.exp(-alpha*(Lenght_of_sample+1))) + B - Last_data_Importance
        return [eq1, eq2]
    # Initial guess for the unknowns
    initial_guess = [1, 1]

    # Solve the system of equations
    solutions = fsolve(equations, initial_guess)

    #print("Solution: ", solutions)
    return solutions[0], solutions[1], alpha

def Log_Coefficient(Last_data_Importance,Lenght_of_sample):
    def equations(vars):
        A,B = vars
        eq1 = A*np.log(1) + B - 1
        eq2 = A*np.log(1+Lenght_of_sample) + B -Last_data_Importance
        return [eq1,eq2]
    initial_guess = [1, 1]

    # Solve the system of equations
    solutions = fsolve(equations, initial_guess)

    #print("Solution: ", solutions)
    return solutions[0], solutions[1]

def Reweighting_p(length_of_time_serise,activation):
    class ActivationFunction:
        @staticmethod
        def sigmoid(x,A,B,alpha):
            return A / (1 + np.exp(-alpha*(x+1))) + B
        @staticmethod
        def log(x,A,B):
            return A*np.log(x+1) + B
        @staticmethod
        def linear(x,A,B):
            return x
    
    af = ActivationFunction()
    x = np.arange(length_of_time_serise)
    if activation == "sigmod":
        A,B, alpha = Signmod_Coefficient(Last_data_Importance=50000, Lenght_of_sample=length_of_time_serise)
        activation_function = af.sigmoid(x,A,B,alpha)
    if activation == "log":
        A, B = Log_Coefficient(Last_data_Importance=50000, Lenght_of_sample=length_of_time_serise)
        activation_function = af.log(x,A,B)
    if activation == "linear":
        A,B = 0,0
        activation_function = af.linear(x, A, B)
    probabilities = activation_function / np.sum(activation_function)
    return probabilities

def Prediction(Time_series,step,N_walk,reweighting_function):
    X1 = Time_series
    #D_X1 = Derivative(Time_series)
    D_X1 = Precentage_Del(Time_series)
    
    Walks = []
    
    if reweighting_function == "none":
        for i in range(N_walk):
            walk = Random_Walk(D_X1, step)
            Walks.append(walk)
    else:
        reweighted_p = Reweighting_p(len(D_X1),reweighting_function)
        for i in range(N_walk):
            walk = Weighted_random_walk(D_X1,reweighted_p, step)
            Walks.append(walk)

    
    Walks = np.array(Walks)
    average_Walk = np.mean(Walks, axis=0) #Checked axis correct
    
    return Walks,average_Walk
    

def Slice_time_series(Time_series,Time_day,T_start,T_end):
    X1 = Time_series
    T = Time_day
    
    # Find the index corresponding to T_start and T_end
    start_index = LD.Find_day_index(T,T_start)
    end_index = LD.Find_day_index(T,T_end)

    # Slicing the arrays from T_start onwards
    Time_string = T[start_index:end_index+1]
    X1_Slicied = X1[start_index:end_index+1]
    
    return Time_string,X1_Slicied
    
def Get_prediction(Time_series,T_data,N_walk,T_Prediction_sample_start,T_Prediction_sample_end,T_Prediction_End,reweighting_function):
    X1 = Time_series
    T = T_data
    #Restict the time series    
    Time_Slicied,X1_Slicied = Slice_time_series(X1,T,T_Prediction_sample_start,T_Prediction_sample_end)
    
    #Set date
    T_prediction,step = Make_days_for_prediction(Time_Slicied[-1],T_Prediction_End)
    
    #Get prediction based on the set date
    if type(T_Prediction_sample_start) == str:
        DX_predictions, DX_prediction_averaged = Prediction(X1,step,N_walk,reweighting_function)
    else:
        start_index = LD.Find_day_index(T,T_Prediction_sample_start)
        end_index = LD.Find_day_index(T,T_Prediction_sample_end)
        restricted_X1 = X1[start_index:end_index]
        DX_predictions, DX_prediction_averaged = Prediction(restricted_X1,step,N_walk,reweighting_function)
    
    #Get average prediction
    X_prediction_ave =[X1_Slicied[-1]]
    for i in range(len(DX_prediction_averaged)-1):
        current_price = X_prediction_ave[i]*(1+DX_prediction_averaged[i])
        X_prediction_ave.append(current_price)
        
    
    #Get prediction from all walks
    X_predictions =[]
    for J in range(N_walk):
        X_prediction =[X1_Slicied[-1]]
        DX_prediction = DX_predictions[J]
        for I in range(len(DX_prediction)-1):
            current_price = X_prediction[I]*(1+DX_prediction[I])
            X_prediction.append(current_price)
        X_predictions.append(X_prediction)
        
    return T_prediction, X_prediction_ave, X_predictions

    
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
    y_pos =  np.mean(error_distibution) # Adjust the y-position of the error bar plot
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
    time_History, price_history = Slice_time_series(X1,T,Plot_from,Plot_to)
    time_History = Convert_str_into_date(time_History)
    time_prediction, X_prediction_ave, X_predictions = Get_prediction(Time_series=X1,Time_day=T,N_walk=N_walk,T_Prediction_sample_start=T_Prediction_sample_start,T_Prediction_sample_end=T_Prediction_sample_end,T_Prediction_End=T_Prediction_End,reweighting_function="none")
    time_prediction, X_prediction_ave_weighted, X_predictions_weighted = Get_prediction(Time_series=X1,Time_day=T,N_walk=N_walk,T_Prediction_sample_start=T_Prediction_sample_start,T_Prediction_sample_end=T_Prediction_sample_end,T_Prediction_End=T_Prediction_End,reweighting_function=reweighting_function)
    
    final_averaged_price = X_prediction_ave[-1]
    final_averaged_price_weighted = X_prediction_ave_weighted[-1]
    
    plot_stock_list = [price_history, X_prediction_ave, X_prediction_ave_weighted]
    plot_days_list = [time_History, time_prediction, time_prediction]
    plot_stock_label = ["history", "simple prediction", "weighted prediction"]
    Plot_stock_colors = ['blue', 'red','green']
    
    # Plot the distribution oif the final position of the walks as an error bar plot
    error_distibution, bp = Generate_error_plot(X_predictions,time_prediction, "red")
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

def plot_true_stock_price(True_stock,start_from):
    
    True_stock_price = True_stock[0]
    T_true = True_stock[1]
    
    start_index_true = LD.Find_day_index(T_true, start_from)
    Time_string_true = T_true[start_index_true:]
    Time_true = Convert_str_into_date(Time_string_true)
    
    Stock_price = True_stock_price[start_index_true:]
    
    plt.plot(Time_true,Stock_price,label="true price",color = "orange")
    
    print("The price at ", Time_string_true[-1], "is ", Stock_price[-1])
    return
    