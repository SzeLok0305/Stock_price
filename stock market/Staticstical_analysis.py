# -*- coding: utf-8 -*-
"""
Created on Tue May 14 18:48:38 2024

@author: User
"""

from Load_data import *

#Close, T_data = Load_Data(FlieLocation="Data/HK0388/past_data/price.csv",ExtractLabel='Close')

#Close_true, T_dat_true = Load_Data(FlieLocation="Data/HK0388/new_data/price.csv",ExtractLabel='Close')

InterestedTimeSeries = 'Close'

History = Load_Data(FlieLocation="Data/HK0388/past_data/price.csv",ExtractLabel=InterestedTimeSeries)
Truth = Load_Data(FlieLocation="Data/HK0388/new_data/price.csv",ExtractLabel=InterestedTimeSeries)


#%% 
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

def Precentage_Del(Time_series,plot=False):
    X1 = Time_series
    D_X1 = []
    D_X1.append(np.float64(0)) # Set initial value
    for i in range(len(X1)-1):
        dx_precentage = (X1[i+1]-X1[i])/X1[i]
        D_X1.append(dx_precentage)
    if plot:
        plt.hist(D_X1,bins="auto",density="True", alpha=1, label='True distribution')
        plt.xlabel("Precentage Price changes")
        plt.title(f'Plot of Delta {Time_series}')
        plt.show()
    
    return D_X1

def Random_Walk(D_Time_series, step):
    # Generate random samples based on the histogram
    num_steps = step  # Define the number of steps in the random walk
    samples = np.random.choice(D_Time_series, size=num_steps, replace=True)
    
    # Perform the random walk
    walk = np.cumsum(samples)
    #check_nan_values(samples)

    # Plot the random walk trajectory
    #plt.plot(walk, label='Random Walk')
    #plt.show()
    return walk

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

def Prediction(Time_series,step,N_walk):
    X1 = Time_series
    #D_X1 = Derivative(Time_series)
    D_X1 = Precentage_Del(Time_series)
    
    Walks = []
    for i in range(N_walk):
        #walk = Random_Walk(D_X1, step)
        walk = Precentage_Random_Walk(D_X1, step)
        Walks.append(walk)
    
    Walks = np.array(Walks)
    average_Walk = np.mean(Walks, axis=0) #Checked axis correct
    
    return Walks,average_Walk
    

def Plot_prediction(Time_series_history,Time_series_true,N_walk,T_start,T_Prediction_sample_start,T_Prediction_sample_end,T_Prediction_End,plot_dis=False):
    """ Time_series_history = time serise of interest in a list:  the data and time """
    """ Time_series_true =  ture value of the time serise of interest in a list:  the data and time """
    """ N_walk = number of random walk"""
    """ T_start = Starting time for plot, in datatime.date"""
    """ T_Prediction_sample_start = Maxmium history to construct the distribution, in datatime.date """
    """ T_Prediction_sample_end = End point of history to construct the distribution, in datatime.date """
    """ T_Prediction_End = predict date """
    
    X1 = Time_series_history[0]
    T = Time_series_history[1]
    
    X1_true = Time_series_true[0]
    T_true = Time_series_true[1]
    
    # Find the index corresponding to T_start and T_end
    date_str = T_start.strftime('%Y-%m-%d')
    start_index = int(np.where(T == T_start.strftime('%Y-%m-%d'))[0])
    
    end_index = int(np.where(T == T_Prediction_sample_end.strftime('%Y-%m-%d'))[0])
    
    

    # Slicing the arrays from T_start onwards
    T_string = T[start_index:end_index]
    X1_Slicied = X1[start_index:end_index]
    # Convert T to matplotlib dates
    T_date = [mdates.datestr2num(str(date)) for date in T_string]
    plt.plot(T_date, X1_Slicied,label="Histroy")
    plt.xlabel("Date")
    plt.ylabel(InterestedTimeSeries)
    plt.title(f'Plot of {InterestedTimeSeries}, HK0388')
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)

    #Find prediction
        
    #Set date
    T_final = datetime.datetime.strptime(T_string[-1], "%Y-%m-%d").date()
    Prediton_date_list = [T_final + datetime.timedelta(days=x) for x in range((T_Prediction_End - T_final).days + 1)] 
    step = len(Prediton_date_list)
    T_prediction = Prediton_date_list
    
    #Get prediction based on the set date
    if type(T_Prediction_sample_start) == str:
        DX_predictions, DX_prediction_averaged = Prediction(X1,step,N_walk)
    else:
        max_date_str = T_Prediction_sample_start.strftime('%Y-%m-%d')
        max_start_index = np.where(T == max_date_str)[0]
        if len(max_start_index) == 0:
            print("No history on this date, please enter another date")
            return  # Stop the function execution
        max_start_index = int(np.where(T == max_date_str)[0])
        restricted_X1 = X1[max_start_index:]
        DX_predictions, DX_prediction_averaged = Prediction(restricted_X1,step,N_walk)
    
    
    #Get average prediction
    X_prediction_ave =[X1[-1]]
    for i in range(len(DX_prediction_averaged)-1):
        current_price = X_prediction_ave[i]*(1+DX_prediction_averaged[i+1])
        X_prediction_ave.append(current_price)
        
    
    #Get prediction from all walks
    X_predictions =[]
    for J in range(N_walk):
        X_prediction =[X1[-1]]
        DX_prediction = DX_predictions[J]
        for I in range(len(DX_prediction)-1):
            current_price = X_prediction[I]*(1+DX_prediction[I+1])
            X_prediction.append(current_price)
        X_predictions.append(X_prediction)
    #X_prediction = DX_prediction_averaged + X1[-1]
    plt.plot(T_prediction, X_prediction_ave,color='r',label="Averaged prediction")
    transparancy = 3/N_walk
    #for K in range(N_walk):
        #plt.plot(T_prediction, X_predictions[K],alpha=transparancy,color='lime')
    plt.ylim(bottom=0)
    plt.axvline(x=T_Prediction_sample_start, linestyle='--', color='r', alpha=0.5, label='Sample Start')
    plt.axvline(x=T_prediction[0], linestyle='--', color="black", alpha=1, label='Sample End')
    
    
    final_averaged_price = X_prediction_ave[-1]
    print("Number of walk is", N_walk)
    print(f"Predicting {InterestedTimeSeries} at", T_Prediction_End)
    print(f"final averaged {InterestedTimeSeries} is", final_averaged_price)
    
    X_predictions_array = np.array(X_predictions)
    # Calculate the 68% and 95% confidence intervals along the 154 axis
    ci_68 = np.percentile(X_predictions_array, q=[16, 84], axis=0)  # 68% cases
    ci_95 = np.percentile(X_predictions_array, q=[2.5, 97.5], axis=0)  # 95% cases
    ci_99 = np.percentile(X_predictions_array, q=[0.5, 99.5], axis=0)  # 95% cases

    plt.plot(T_prediction, ci_68[0],alpha=0.7,color='aqua',label="68% cases")
    plt.plot(T_prediction, ci_68[1],alpha=0.7,color='aqua')
    plt.plot(T_prediction, ci_99[0],alpha=0.7,color='lime',label="99% cases")
    plt.plot(T_prediction, ci_99[1],alpha=0.7,color='lime')
    
    # True data (from future)
    date_str_true = T_string[-1]
    start_index_true = int(np.where(T_true == date_str_true)[0])
    T_string_true = T_true[start_index_true:]
    T_date_new = [mdates.datestr2num(str(date)) for date in T_string_true]
    plt.plot(T_date_new,X1_true[start_index_true:], label="true data")
    
    
    plt.legend()
    plt.show()
    
    print("68% of simulation fall in range ", ci_68[0][-1], ci_68[1][-1])
    print("95% of simulation fall in range ", ci_95[0][-1], ci_95[1][-1])
    print("99% of simulation fall in range ", ci_99[0][-1], ci_99[1][-1])
    
    print(f"True {InterestedTimeSeries} is", X1_true[-1])
    
    if plot_dis:
        
        CombinDxOverAllWalk = DX_predictions.flatten()
        
        plt.hist(CombinDxOverAllWalk,bins="auto",density="True", alpha=1, label='True distribution')
        plt.xlabel("% changes in price")
        plt.title(f'Delta {InterestedTimeSeries}, Combine all walk')
    
        # Get the current plot limits
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        # Calculate the coordinates for the text
        text_x = x_max - 0.6 * (x_max - x_min)
        text_y = y_max - 0.1 * (y_max - y_min)
        plt.text(text_x, text_y, f"Considered histroy from {max_date_str}")
        plt.show()
        
        for count_I in range(len(DX_prediction_averaged)):
            
            plt.hist(DX_predictions[count_I],bins="auto", alpha=0.5)
        plt.xlabel("% changes in price")
        plt.title(f'Delta {InterestedTimeSeries}, overlap all walk')
        # Get the current plot limits
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        # Calculate the coordinates for the text
        text_x = x_max - 0.6 * (x_max - x_min)
        text_y = y_max - 0.1 * (y_max - y_min)
        plt.text(text_x, text_y, f"Considered histroy from {max_date_str}")
        plt.show()
        

    return DX_predictions,X_prediction_ave,X_predictions

Plot_Starting_date = datetime.date(2015,8,4)
Prediction_Ending_date = datetime.date(2024,5,20)
Sample_start_date=datetime.date(2017,8,4)
Sample_end_date=datetime.date(2023,8,4)

XX,YY,ZZ = Plot_prediction(Time_series_history=History,Time_series_true=Truth,N_walk=10000,T_start=Plot_Starting_date,T_Prediction_sample_start=Sample_start_date,T_Prediction_sample_end=Sample_end_date,T_Prediction_End=Prediction_Ending_date,plot_dis=False)

