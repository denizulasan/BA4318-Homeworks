# 1-a
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    rolmean = pd.Series(timeseries).rolling(window=1440).mean()
    rolstd = pd.Series(timeseries).rolling(window=1440).std()
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    print("Results of Dickey-Fuller Test:")
    array = np.asarray(timeseries, dtype='float')
    np.nan_to_num(array,copy=False)
    dftest = adfuller(array, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

# Load data
df = pd.read_csv("HW03_USD_TRY_Trading.txt", sep='\t')

seriesname = 'Close'
series = df[seriesname]

test_stationarity(series)  #For entire period stationarity 

size = len(series)
train = series[0:10079]  # Until May 6, there are 6 days + 23 Hours + 59 minutes = 10079 min  
test = series [10079:]
testarray = np.asarray(test, dtype=float)
test_stationarity(testarray)           # For May 6 stationarity

#1-b
from math import sqrt
import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing, Holt   
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2))


name = 'Close'
freq = 956 # May 6, 16 saat - 4 minutes
series = df[name]
numbers = np.asarray(series,dtype='int')
result = sm.tsa.seasonal_decompose(numbers,freq=956,model='Additive')
result.plot()
plt.show() # Uncomment to reshow plot, saved as Figure 1. 

#May 6 15.57 estimation

print("Estimations for 15.57")
# Function for Naive
def estimate_naive(df, seriesname):
     numbers = np.asarray ( df[seriesname] )
     return float( numbers[-2] )
naive = round(estimate_naive (df.iloc[10082:11038], seriesname),4)
print ("Naive for 15.57:", naive)
RMSE = rmse(naive, 6.01850)
print("RMSE Naive for 15.57: ", RMSE)


# Function for Simple Average
def estimate_simple_average(df,seriesname):
    avg = df[seriesname].mean()
    return avg

simpleaverage = round(estimate_simple_average(df.iloc[10082:11038], seriesname), 4)
print("Simple average estimation:", simpleaverage)
RMSE = rmse(simpleaverage, 6.01850)
print("RMSE simpleaverage for 15.57: ", RMSE)

# Function for Moving Average
def estimate_moving_average(df,seriesname,windowsize):
    avg = df[seriesname].rolling(windowsize).mean().iloc[-1]
    return avg

periods = 60
movingaverage = round(estimate_moving_average(df.iloc[10082:11038],seriesname, periods),4)

print("Moving average for 15.57:", movingaverage)
RMSE = rmse(movingaverage, 6.01850)
print("RMSE movingaverage for 15.57: ", RMSE)

# Function for Simple Exponential Smoothing
def estimate_ses(df, seriesname, alpha=0.2):
    numbers = np.asarray(df[seriesname])
    estimate = SimpleExpSmoothing(numbers).fit(smoothing_level=alpha,optimized=False).forecast(1)
    return estimate

alpha = 0.2
ses = round ( estimate_ses(df.iloc[10082:11038], seriesname, alpha)[0], 4)
print("Exponential smoothing estimation with alpha =", alpha, ": ", ses)
RMSE = rmse(ses, 6.01850)
print("RMSE simple exponential smoothing for 15.57: ", RMSE)

# Trend estimation with Holt
def estimate_holt(df, seriesname, alpha=0.2, slope=0.1):
    numbers = np.asarray(df[seriesname])
    model = Holt(numbers)
    fit = model.fit(alpha,slope)
    estimate = fit.forecast(1)[-1]
    return estimate

alpha = 0.2
slope = 0.1
holt = round(estimate_holt(df.iloc[10082:11038],seriesname,alpha, slope),4)
print("Holt trend estimation for 15.57 with alpha =", alpha, ", and slope =", slope, ": ", holt)
RMSE = rmse(holt, 6.01850)
print("RMSE holt for 15.57: ", RMSE)
# Trend and seasonality estimation with Holt-Winters ___ I could not do it.


# Estimation for 15.58
print("Estimations for 15.58")
freq = 957 # May 6, 16 saat - 3 minutes

# for Naive

naive = round(estimate_naive (df.iloc[10083:11039], seriesname),4)
print ("Naive for 15.58:", naive)
RMSE = rmse(naive, 6.01933)
print("RMSE Naive for 15.58: ", RMSE)


#  for Simple Average

simpleaverage = round(estimate_simple_average(df.iloc[10083:11039], seriesname), 4)
print("Simple average estimation:", simpleaverage)
RMSE = rmse(simpleaverage, 6.01933)
print("RMSE simpleaverage for 15.58: ", RMSE)

# for Moving Average

minutes = 60
movingaverage = round(estimate_moving_average(df.iloc[10083:11039],seriesname, minutes),4)
print("Moving average for 15.58:", movingaverage)
RMSE = rmse(movingaverage, 6.01933)
print("RMSE movingaverage for 15.58: ", RMSE)

# for Simple Exponential Smoothing

alpha = 0.2
ses = round ( estimate_ses(df.iloc[10083:11039], seriesname, alpha)[0], 4)
print("Exponential smoothing estimation with alpha =", alpha, ": ", ses)
RMSE = rmse(ses, 6.01933)
print("RMSE simple exponential smoothing for 15.57: ", RMSE)

# Trend estimation with Holt

alpha = 0.2
slope = 0.1
holt = round(estimate_holt(df.iloc[10083:11039],seriesname,alpha, slope),4)
print("Holt trend estimation for 15.57 with alpha =", alpha, ", and slope =", slope, ": ", holt)
RMSE = rmse(holt, 6.01933)
print("RMSE holt for 15.57: ", RMSE)

# Smallest RMSE is moving average, so I estimated 15.59 and 16.00 with this metmod.
# Since Simple Moving Average (SMA) takes the average over some set number of time periods, It will be same for any future period. Thus, in the long run, it may not be a good indicator.
minutes = 60
movingaverage = round(estimate_moving_average(df.iloc[10083:11039],seriesname, minutes),4)
print("Moving average for 15.59:", movingaverage)

minutes = 60
movingaverave = round(estimate_moving_average(df.iloc[10083:11039],seriesname, minutes),4)
print("Moving average for 16.00:", movingaverage)

minutes = 60
movingaverage = round(estimate_moving_average(df.iloc[10083:11039],seriesname, minutes),4)
print("Moving average for May 7, 16.00:", movingaverage)

#1-c
df2 = df.dropna()
df2.info()
print("estimations without zeros and missing values")

seriesname = 'Close'
series = df2[seriesname]
test_stationarity(series)
size = len(series)
train = series[0:10079]  # Until May 6, there are 6 days + 23 Hours + 59 minutes = 10079 min  
test = series [10079:]
testarray = np.asarray(test, dtype=float)
test_stationarity(testarray)           # For May 6 stationarity

print("Estimations for 15.57")
# Function for Naive
naive = round(estimate_naive (df2.iloc[10082:11038], seriesname),4)
print ("Naive for 15.57:", naive)
RMSE = rmse(naive, 6.01850)
print("RMSE Naive for 15.57: ", RMSE)

# Function for Simple Average
simpleaverage = round(estimate_simple_average(df2.iloc[10082:11038], seriesname), 4)
print("Simple average estimation:", simpleaverage)
RMSE = rmse(simpleaverage, 6.01850)
print("RMSE simpleaverage for 15.57: ", RMSE)

# Function for Moving Average

periods = 60
movingaverage = round(estimate_moving_average(df2.iloc[10082:11038],seriesname, periods),4)

print("Moving average for 15.57:", movingaverage)
RMSE = rmse(movingaverage, 6.01850)
print("RMSE movingaverage for 15.57: ", RMSE)

# Function for Simple Exponential Smoothing

alpha = 0.2
ses = round ( estimate_ses(df2.iloc[10082:11038], seriesname, alpha)[0], 4)
print("Exponential smoothing estimation with alpha =", alpha, ": ", ses)
RMSE = rmse(ses, 6.01850)
print("RMSE simple exponential smoothing for 15.57: ", RMSE)

# Trend estimation with Holt


alpha = 0.2
slope = 0.1
holt = round(estimate_holt(df2.iloc[10082:11038],seriesname,alpha, slope),4)
print("Holt trend estimation for 15.57 with alpha =", alpha, ", and slope =", slope, ": ", holt)
RMSE = rmse(holt, 6.01850)
print("RMSE holt for 15.57: ", RMSE)
# Trend and seasonality estimation with Holt-Winters ___ I could not do it.


# Estimation for 15.58
print("Estimations for 15.58")
freq = 957 # May 6, 16 saat - 3 minutes

# for Naive

naive = round(estimate_naive (df2.iloc[10083:11039], seriesname),4)
print ("Naive for 15.58:", naive)
RMSE = rmse(naive, 6.01933)
print("RMSE Naive for 15.58: ", RMSE)


#  for Simple Average

simpleaverage = round(estimate_simple_average(df2.iloc[10083:11039], seriesname), 4)
print("Simple average estimation:", simpleaverage)
RMSE = rmse(simpleaverage, 6.01933)
print("RMSE simpleaverage for 15.58: ", RMSE)

# for Moving Average

minutes = 60
movingaverage = round(estimate_moving_average(df2.iloc[10083:11039],seriesname, minutes),4)
print("Moving average for 15.58:", movingaverage)
RMSE = rmse(movingaverage, 6.01933)
print("RMSE movingaverage for 15.58: ", RMSE)

# for Simple Exponential Smoothing

alpha = 0.2
ses = round ( estimate_ses(df2.iloc[10083:11039], seriesname, alpha)[0], 4)
print("Exponential smoothing estimation with alpha =", alpha, ": ", ses)
RMSE = rmse(ses, 6.01933)
print("RMSE simple exponential smoothing for 15.57: ", RMSE)

# Trend estimation with Holt

alpha = 0.2
slope = 0.1
holt = round(estimate_holt(df2.iloc[10083:11039],seriesname,alpha, slope),4)
print("Holt trend estimation for 15.57 with alpha =", alpha, ", and slope =", slope, ": ", holt)
RMSE = rmse(holt, 6.01933)
print("RMSE holt for 15.57: ", RMSE)

# Smallest RMSE is moving average, so I estimated 15.59 and 16.00 with this metmod.
# Since Simple Moving Average (SMA) takes the average over some set number of time periods, It will be same for any future period. Thus, in the long run, it may not be a good indicator.
minutes = 60
movingaverage = round(estimate_moving_average(df2.iloc[10083:11039],seriesname, minutes),4)
print("Moving average for 15.59:", movingaverage)

minutes = 60
movingaverave = round(estimate_moving_average(df2.iloc[10083:11039],seriesname, minutes),4)
print("Moving average for 16.00:", movingaverage)

minutes = 60
movingaverage = round(estimate_moving_average(df2.iloc[10083:11039],seriesname, minutes),4)
print("Moving average for May 7, 16.00:", movingaverage)

# Question 2

df3 = df.loc[df['Time'].isin(['00:57','01:57','02:57','03:57','04:57','05:57','06:57','07:57','08:57','09:57','10:57','11:57','12:57',
                                  '13:57','14:57','15:57','16:57','17:57','18:57','19:57','20:57','21:57','22:57','23:57',
                                  '00:58','01:58','02:58','03:58','04:58','05:58','06:58','07:58','08:58','09:58','10:58','11:58','12:58',
                                  '13:58','14:58','15:58','16:58','17:58','18:58','19:58','20:58','21:58','22:58','23:58',])]

print(df3)
# 2-a
seriesname = 'Close'
series = df3[seriesname]

test_stationarity(series)

# 2-b
print("Estimations for 15.57 with end times")


# Function for Naive
seriesname = 'Close'


naive = round(estimate_naive (df3.iloc[10197:10978], seriesname),4)
print ("Naive for 15.57:", naive)
RMSE = rmse(naive, 6.01850)
print("RMSE Naive for 15.57: ", RMSE)

# Function for Simple Average
simpleaverage = round(estimate_simple_average(df3.iloc[10197:10978], seriesname), 4)
print("Simple average estimation:", simpleaverage)
RMSE = rmse(simpleaverage, 6.01850)
print("RMSE simpleaverage for 15.57: ", RMSE)

# Function for Moving Average

periods = 60
movingaverage = round(estimate_moving_average(df3.iloc[10197:10978],seriesname, periods),4)

print("Moving average for 15.57:", movingaverage)
RMSE = rmse(movingaverage, 6.01850)
print("RMSE movingaverage for 15.57: ", RMSE)

# Function for Simple Exponential Smoothing

alpha = 0.2
ses = round ( estimate_ses(df3.iloc[10197:10978], seriesname, alpha)[0], 4)
print("Exponential smoothing estimation with alpha =", alpha, ": ", ses)
RMSE = rmse(ses, 6.01850)
print("RMSE simple exponential smoothing for 15.57: ", RMSE)

# Trend estimation with Holt


alpha = 0.2
slope = 0.1
holt = round(estimate_holt(df3.iloc[10197:10978],seriesname,alpha, slope),4)
print("Holt trend estimation for 15.57 with alpha =", alpha, ", and slope =", slope, ": ", holt)
RMSE = rmse(holt, 6.01850)
print("RMSE holt for 15.57: ", RMSE)
# Trend and seasonality estimation with Holt-Winters ___ I could not do it.


# Estimation for 15.58
print("Estimations for 15.58 with end times")

size = len(series)
train = df3[:200]
test = df3[200:]
testarray = np.asarray(test.Close)

# for Naive
test['naive'] = testarray[len(testarray)-1]
naive = round(estimate_naive (df3.iloc[10197:11037], seriesname),4)
print ("Naive for 15.58:", naive)
RMSE = rmse(naive, 6.01933)
print("RMSE Naive for 15.58: ", RMSE)


#  for Simple Average

simpleaverage = round(estimate_simple_average(df3.iloc[10197:11037], seriesname), 4)
print("Simple average estimation:", simpleaverage)
RMSE = rmse(simpleaverage, 6.01933)
print("RMSE simpleaverage for 15.58: ", RMSE)

# for Moving Average

minutes = 60
movingaverage = round(estimate_moving_average(df3.iloc[10197:11037],seriesname, minutes),4)
print("Moving average for 15.58:", movingaverage)
RMSE = rmse(movingaverage, 6.01933)
print("RMSE movingaverage for 15.58: ", RMSE)

# for Simple Exponential Smoothing

alpha = 0.2
ses = round ( estimate_ses(df3.iloc[10197:11037], seriesname, alpha)[0], 4)
print("Exponential smoothing estimation with alpha =", alpha, ": ", ses)
RMSE = rmse(ses, 6.01933)
print("RMSE simple exponential smoothing for 15.57: ", RMSE)

# Trend estimation with Holt

alpha = 0.2
slope = 0.1
holt = round(estimate_holt(df3.iloc[10197:11037],seriesname,alpha, slope),4)
print("Holt trend estimation for 15.57 with alpha =", alpha, ", and slope =", slope, ": ", holt)
RMSE = rmse(holt, 6.01933)
print("RMSE holt for 15.57: ", RMSE)

# Smallest RMSE is moving average, so I estimated 15.59 and 16.00 with this metmod.
# Since Simple Moving Average (SMA) takes the average over some set number of time periods, It will be same for any future period. Thus, in the long run, it may not be a good indicator.
minutes = 60
movingaverage = round(estimate_moving_average(df3.iloc[1197:11037],seriesname, minutes),4)
print("Moving average for 15.59:", movingaverage)

minutes = 60
movingaverave = round(estimate_moving_average(df3.iloc[10197:11037],seriesname, minutes),4)
print("Moving average for 16.00:", movingaverage)

minutes = 60
movingaverage = round(estimate_moving_average(df3.iloc[10197:11037],seriesname, minutes),4)
print("Moving average for May 7, 16.00:", movingaverage)

#2-c

df4 = df3.dropna()
df4.info()
print("estimations without missing values")

seriesname = 'Close'
series = df4[seriesname]
test_stationarity(series)

# Question 3
df = df.dropna()
df= df[df['Volume'] != 0]
seriesname = 'Close'
series = df[seriesname]

# 3-a
test_stationarity(series)

def estimate_moving_average(df,seriesname,windowsize):
    avg = df[seriesname].rolling(windowsize).mean().iloc[-1]
    return avg

periods = 60
movingaverage = round(estimate_moving_average(df.iloc[10082:11038],seriesname, periods),4)
print("Moving average for 15.57:", movingaverage)
RMSE = rmse(movingaverage, 6.01850)
print("RMSE movingaverage for 15.57: ", RMSE)

movingaverage = round(estimate_moving_average(df.iloc[10083:11039],seriesname, minutes),4)
print("Moving average for 15.58:", movingaverage)
RMSE = rmse(movingaverage, 6.01933)
print("RMSE movingaverage for 15.58: ", RMSE)


movingaverage = round(estimate_moving_average(df.iloc[10083:11039],seriesname, minutes),4)
print("Moving average for 15.59:", movingaverage)

movingaverave = round(estimate_moving_average(df.iloc[10083:11039],seriesname, minutes),4)
print("Moving average for 16.00:", movingaverage)

movingaverage = round(estimate_moving_average(df.iloc[10083:11039],seriesname, minutes),4)
print("Moving average for May 7, 16.00:", movingaverage)
#3-b

def linear_weight_moving_average(signal, period):
    buffer = [np.nan] * period
    for i in range(period, len(signal)):
        buffer.append(
            (signal[i - period : i] * (np.arange(period) + 1)).sum()
            / (np.arange(period) + 1).sum()
        )
    return buffer

print("linear moving average:", linear_weight_moving_average(seriesname, period=60))

#3-c
Random = df.sample()
print("Random time is", Random['Time'],"and the Close value at that time is", Random['Close'])






