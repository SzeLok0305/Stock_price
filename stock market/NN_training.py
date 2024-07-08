import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import matplotlib.pyplot as plt
#%%
""""      The objective is to identify buying/selling oppotunities   """


# criteria for buying/selling
# 1:If today price is higher than the moving average by N-sigma, then I sell
# 2:If today price is lower than the moving average by N-sigma, then I buy

def Moving_average_volatility_order(Time_series, averaging_days, volatility_order):
    label = []
    for i in range(len(Time_series)):
        start_index = max(i - averaging_days + 1, 0)
        average = sum(Time_series[start_index:i+1]) / (i - start_index + 1)
        volatility = torch.std(Time_series[start_index:i+1])
        #How many sigma larger then the averge would I sell it
        threshold_plus = average + volatility_order * volatility
        threshold_minus = average - volatility_order * volatility
        #print(start_index)
        #print(Time_series[i])
        #print(average,volatility)

        if Time_series[i] > threshold_plus:
            label.append(0)  # Sell the stock
        elif Time_series[i] < threshold_minus:
            label.append(1)  # buy the stock
        else:
            label.append(1)  # neutral

    return torch.tensor(label, dtype=torch.float).view(-1, 1)

def Open_Close_Comparasion(Open,Close):
    
    label=[]
    for i in range(len(Open)):
        if Close[i] >= Open[i]:
            label.append(1) #buy
        elif Close[i] < Open[i]:
            label.append(0) #short
            
    return torch.tensor(label, dtype=torch.float).view(-1, 1)

def Moving_average_Gaussian(Close,averaging_days):
    labels=[]
    for i in range(len(Close)):
        start_index = max(i - averaging_days + 1, 0)
        average = sum(Close[start_index:i+1]) / (i - start_index + 1)
        volatility = torch.std(torch.tensor(Close[start_index:i+1]))
        
        # Calculate the CDF value of current 'Close' based on Gaussian distribution
        cdf_value = 0.5 * (1 + torch.erf((Close[i] - average) / (volatility * torch.sqrt(torch.tensor(2.)))))
        labels.append(cdf_value)
    
    return torch.tensor(labels, dtype=torch.float).view(-1, 1)

def Check_label_distribution(label):
    plt.hist(label,bins="auto",density="True")
    plt.xlabel("Label values")
    plt.show()
    return
            
class Model_structure(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model_structure, self).__init__()
        
        #types of layer
        hidden_size = 25
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fcFinal = nn.Linear(hidden_size, output_size)
        
        #types of activation function
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.4)
        self.sigmod = nn.Sigmoid()
        self.normalize1D = nn.BatchNorm1d(num_features=1)
        self.normalize2D = nn.BatchNorm1d(num_features=2)

    def forward(self, x):
        # Separate the tensor
        #tensor_small = x[:,:2]
        out = x
        #tensor_big = x[:,2:]

        # Normalize separately
        #out = self.normalize3D(tensor_small)
        #out2 = self.normalize1D(tensor_big)

        # Concatenate the tensors along the second dimension
        #x_p = torch.cat((out1, out2), dim=1)
        
        out = self.fc1(out)
        out = self.drop(out)
        out = self.sigmod(out)
        
        out = self.fc2(out)
        out = self.drop(out)
        out = self.sigmod(out)
        
        out = self.fc3(out)
        out = self.drop(out)
        out = self.relu(out)
        
        out = self.fcFinal(out)
        out = self.relu(out)
        return out

def Train_Model(stock_name,label_model):
#Load Data
    import Load_data as LD
    History = LD.Load_mutiple_data(FlieLocation="Stock_Data/" + stock_name+ "/Historical_data/past_data/price.csv",ExtractLabels=["Open","Close","Volume"])

    Open = torch.tensor(History[0][0][0]).view(-1, 1).float()
    Close = torch.tensor(History[0][1][0]).view(-1, 1).float()
    #Volume = torch.tensor(History[0][2][0]).view(-1, 1).float()
    
    #Train_label = Train_data_and_Label(Open,50,2)
    #Train_label = Labeling_stretagy(Open,Close)
    
    open_yestaday = Open[:-1]
    
    close_yestaday = Close[:-1]
    
    open_today = Open[1:]
    
    Train_data = torch.cat((open_yestaday, close_yestaday, open_today), dim=1)

    Last_element = torch.cat((open_yestaday[-1:], close_yestaday[-1:], open_today[-1:]), dim=1)
    
    Train_data = torch.cat((open_yestaday, close_yestaday, open_today), dim=1)
    
    Averaging_day = 50
    #label_model = "Moving_average_Gaussian"
    
    if label_model == "Moving_average_Gaussian":
        Train_label = Moving_average_Gaussian(Close,Averaging_day) #N-days moving mean
        model_name = str(Averaging_day) + "_days.pth"
    elif label_model == "Open_Close_Comparasion":
        Train_label = Open_Close_Comparasion(Open,Close)
        model_name = "OC.pth"
    elif label_model == "Moving_average_volatility_order":
        VO = 1
        Train_label = Moving_average_volatility_order(Open,Averaging_day,VO)
        model_name = "VO" + str(VO) + "_" + str(Averaging_day) + "_days.pth"
    else:
        print("Please choose a labeling strategy!")
        return
        
        
    
    Train_label = Train_label[1:] #Remove the first element
    Check_label_distribution(Train_label.view(-1))

    
    # Instantiate your custom network
    input_size = Train_data.shape[1]#*Train_data.shape[0]
    output_size = Train_label.shape[1]
    model = Model_structure(input_size, output_size)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize a list to store the loss values
    loss_values = []

    # Set the number of training epochs
    num_epochs = 2**13

    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(Train_data)
        loss = criterion(outputs, Train_label)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store the loss value for this epoch
        loss_values.append(loss.item())

        # Print the training loss for every few epochs
        if (epoch+1) % 10 == 0:
            print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    print("total time used is ", time.time() - start_time, "seconds")
    print("Which is, " ,(time.time() - start_time)/3600, "hours")
    # Save the trained model weights
    save_dir = "Stock_Data/" + stock_name + "/trained_weights/" + label_model + "/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), save_path)

    # Plotting the error function against the epoch
    plt.plot(range(1, num_epochs+1), loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    
    return model,Last_element

def Train_all_model():
    #model_variation = ["Moving_average_Gaussian","Open_Close_Comparasion","Moving_average_volatility_order"]
    model_variation = ["Moving_average_Gaussian","Moving_average_volatility_order"]
    for mv in model_variation:
        Train_Model("NVDA",mv)
    return
    
def Generate_scorce(model,features):
    model.eval()
    scorce = model(features)
    count = np.count_nonzero(scorce < 0)
    if count > 0:
        print("Negative value found in NN output")
    return scorce
#%%
def Evaluate_model(model,Last_element):
    
    # Set the model to evaluation mode
    model.eval()

    # Make predictions
    predictions = []
    x_test = torch.linspace(0, 2, 10)
    with torch.no_grad():
        for x in x_test:
            test_input = Last_element*x
            print(test_input.size())
            prediction = model(test_input)
            predictions.append(prediction)


    predictions = torch.tensor(predictions)
    
    plt.plot(x_test,predictions)
    plt.ylabel("selling scorce")

    return
#Evaluate_model(model,Last_element)

#%%

def Load_model(stock_name, Model_ver):
    my_model = Model_structure(3, 1)
    print("Using model:", Model_ver)
    Prefix_dir = "Stock_Data/" + stock_name + "/trained_weights/"
    
    my_model.load_state_dict(torch.load(Prefix_dir+ Model_ver + ".pth"))
    return my_model









