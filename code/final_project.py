import os
import statsmodels.api as sm
import pandas as pd
from requests import get
from io import BytesIO
from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from functions import *
from pylab import rcParams
# from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
import statsmodels.tsa.holtwinters as ets
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
register_matplotlib_converters()


# Objective
# %%------------------------------------------------------------------------------------------------------------
print("To predict the RELATIVE HUMIDITY of a given point of time based on the all other "
      "attributes affecting the change in RH")


# LOAD THE DATASET
# %%------------------------------------------------------------------------------------------------------------
print('\n')
print(20 * "-" + "loading dataset..." + 20 * "-")

if "AirQualityUCI" not in os.listdir():
    request = get('https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip')
    zip_file = ZipFile(BytesIO(request.content))
    zip_file.extractall()
print('\n')
df = pd.read_csv("AirQualityUCI.csv", sep = ';',infer_datetime_format=True)
print(20 * "-" + "Download Complete!" + 20 * "-")


# PREPROCESSING
# %%------------------------------------------------------------------------------------------------------------
print('\n')
print(20 * "-" + "PREPROCESSING" + 20 * "-")
df.describe()

# Investigate the first 5 rows
print(df.head(5))

# Removing last two Unnamed columns
df = df.drop(['Unnamed: 15','Unnamed: 16'], axis = 1)


# Summary of dataframe
print('Summary of Dataframe:\n',df.info)


# Changing the datatype from object to float
print('Datatypes in Dataframe\n',df.dtypes)
df['CO(GT)'] = df['CO(GT)'].str.replace(',', '.').astype(float)
df['C6H6(GT)'] = df['C6H6(GT)'].str.replace(',','.').astype(float)
df['T'] = df['T'].str.replace(',', '.').astype(float)
df['RH'] = df['RH'].str.replace(',', '.').astype(float)
df['AH'] = df['AH'].str.replace(',', '.').astype(float)


print('\n')
# Dimension of the Dataset
print('Shape of the Dataset before null value removal:',df.shape)
# print('Shape of the Dataset:\n',df.shape)

# Handling null values
print('\n')
# Estimating null values
print(df.isnull().sum())

# Removing null values
null_data = df[df.isnull().any(axis=1)]
print('\n',null_data.head())
df= df.dropna()
# print(df.head)
print('Shape of the Dataset after null value removal',df.shape)

# Replacing -200 with nan
df = df.replace(-200,np.nan)
print(df.isnull().sum())

# Appending date and time
print(df.index)
df.loc[:,'Datetime'] = df['Date'] + ' ' + df['Time']
DateTime = []
for x in df['Datetime']:
    DateTime.append(datetime.strptime(x,'%d/%m/%Y %H.%M.%S'))
datetime = pd.Series(DateTime)
df.index = datetime
# print(df.head())
# print('AFTER',df.dtypes)
df = df.replace(-200, np.nan)

# Handling Nan values
# Creating processed dataframe
print(df.isnull().sum())
print(df.head)
# SD = df['Date']
# ST = df['Time']
S0 = df['CO(GT)'].fillna(df['PT08.S1(CO)'].mean())
S1 = df['PT08.S1(CO)'].fillna(df['PT08.S1(CO)'].mean())
S2 = df['NMHC(GT)'].fillna(df['NMHC(GT)'].mean())
S3 = df['C6H6(GT)'].fillna(df['C6H6(GT)'].mean())
S4 = df['PT08.S2(NMHC)'].fillna(df['PT08.S1(CO)'].mean())
S5 = df['NOx(GT)'].fillna(df['NOx(GT)'].mean())
S6 = df['PT08.S3(NOx)'].fillna(df['PT08.S1(CO)'].mean())
S7 = df['NO2(GT)'].fillna(df['NO2(GT)'].mean())
S8 = df['PT08.S4(NO2)'].fillna(df['PT08.S1(CO)'].mean())
S9 = df['PT08.S5(O3)'].fillna(df['PT08.S1(CO)'].mean())
S10 = df['T'].fillna(df['T'].mean())
S11 = df['RH'].fillna(df['RH'].mean())
S12 = df['AH'].fillna(df['AH'].mean())
print('Handling nan with mean\n',df.isnull().sum())
print('\n')

df = pd.DataFrame({'CO(GT)':S0,'PT08.S1(CO)':S1,'NMHC(GT)':S2, 'C6H6(GT)':S3, 'PT08.S2(NMHC)':S4, 'NOx(GT)':S5,
                   'PT08.S3(NOx)':S6, 'NO2(GT)':S7,  'PT08.S4(NO2)':S8, 'PT08.S5(O3)':S9, 'T':S10, 'RH':S11, 'AH':S12 })

print("cleaned dataset after preprocessing:\n", df)
print(df.shape)
# df.index = datetime
# df.to_csv("AQI.csv")
# print('CLEANDED DATASET:\n')
# print('CLEANDED DATASET:\n',df.head)

# Section 5: Description of the dataset. Describe the independent variable(s) and dependent variable:
# %%------------------------------------------------------------------------------------------------------------

# split into train and test(20%) dataset
train, test = split_df_train_test(df, 0.2)
print()

print("The dimension of train data is:")
print(train.shape)

# dimension of test data
print()

print("The dimension of test data is:")
print(test.shape)


# Created Dataframe for Dependent variable and time
df_rh = pd.DataFrame({'RH':S11})

# df.to_csv("AirQuality_processed_rh.csv")
print('Dataframe for Dependent variable and time\n',df_rh.head())

# dependent variable v/s time
plt.figure(figsize=(16,10))
plt.plot(df_rh,  label = 'RH')
plt.xlabel('Time: March 2004- February 2005')
plt.ylabel('Relative Humidity (RH)')
plt.title('Time Series plot of Relative humidity')
plt.legend(loc='best')
plt.show()

# ACF of the dependent variable.
autocorrelation = cal_auto_correlation(list(df["RH"]), 200)
plot_acf(autocorrelation, "ACF plot for Relative Humidity")


# auto corrleation using custom function
y = df['RH']
k = 20
acfcal = auto_corr_cal(y,k)
acfplotvals = acfcal[::-1] + acfcal[1:]
plt.figure(figsize=(16,10))
plt.stem(range(-(k - 1), k), acfplotvals)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('ACF using custom function 1')
plt.show()

# heatmap
corr = df.corr()
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, square=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
rcParams['figure.figsize'] = 16, 10
plt.show()

# section 6 : #Stationarity
# %%------------------------------------------------------------------------------------------------------------
print('\n')
test_result = adfuller(df['RH'])
adfuller_test(df['RH'])

# section 7 : # Time series decomposition
# %%------------------------------------------------------------------------------------------------------------
from pylab import rcParams
rcParams['figure.figsize'] = 16, 10
decomposition = sm.tsa.seasonal_decompose(train["RH"], model='additive')
fig = decomposition.plot()
plt.title('Additive Residuals')
plt.show()

rcParams['figure.figsize'] = 16, 10
decomposition = sm.tsa.seasonal_decompose(train["RH"], model='multiplicative')
fig = decomposition.plot()
plt.title('Multiplicative Residuals')
plt.show()



y = df['RH'].astype(float)
print(y)
STL = STL(y)
res = STL.fit()
fig = res.plot()
# plt.fig(figsize=(16,10))
plt.show()

T = res.trend
S = res.seasonal
R = res.resid

plt.figure(figsize=(16,10))
plt.plot(T, label='trend')
plt.plot(S, label='Seasonal')
plt.plot(R, label='residuals')
plt.xlabel('Year')
plt.ylabel('Magnitude')
plt.title('Trend, Seasonality, Residual components using STL Decomposition')
plt.legend()
plt.show()

adjusted_seasonal = y-S
plt.figure(figsize=(16,10))
plt.plot(y[:50], label='Original')
plt.plot(adjusted_seasonal[:50], label='Seasonally Adjusted')
plt.xlabel('Date')
plt.ylabel('Magnitude')
plt.title('Original vs Seasonally adjusted')
plt.legend()
plt.show()

# Measuring strength of trend and seasonality
F = np.max([0,1-np.var(np.array(R))/np.var(np.array(T+R))])
print('Strength of trend for Air quality dataset is', round(F,3))

FS = np.max([0, 1-np.var(np.array(R))/np.var(np.array(S+R))])
print('Strength of seasonality for Air quality dataset is', round(FS,3))



# section 8 : # Base Models
# %%------------------------------------------------------------------------------------------------------------
# #performance for all the models
result_performance = pd.DataFrame(
        {"Model": [], "MSE": [], "RMSE": [], "Residual Mean": [], "Residual Variance": []})


print(20 * "-" + "AVERAGE MODEL" + 20 * "-")
# %%------------------------------------------------------------------------------------------------------------
average_predictions = generic_average_method(train["RH"], len(test["RH"]))
print(average_predictions)
print(test["RH"])
x = test["RH"] - average_predictions
print(x)

avg_mse = cal_mse(test["RH"], average_predictions)
print()
print("The MSE for Average model is:")
print(avg_mse)


avg_rmse = np.sqrt(avg_mse)
print()
print("The RMSE for Average model is:")
print(avg_rmse)

# forecast errors for average model
residuals_avg = cal_forecast_errors(test["RH"], average_predictions)

# average residual variance
avg_variance = np.var(residuals_avg)
print()
print("The Variance of residual for Average model is:")
print(avg_variance)

# Average residual mean
avg_mean = np.mean(residuals_avg)
print()
print("The Mean of residual for Average model is:")
print(avg_mean)

# Average residual ACF
residual_autocorrelation_average = cal_auto_correlation(residuals_avg, len(average_predictions))
plot_acf(residual_autocorrelation_average, "ACF plot using Average Residuals")


# add the results to common dataframe
result_performance = result_performance.append(
pd.DataFrame(
            {"Model": ["Average Model"], "MSE": [avg_mse], "RMSE": [avg_rmse],
             "Residual Mean": [avg_mean], "Residual Variance": [avg_variance]}))

# plot the predicted vs actual data
average_df = test.copy(deep=True)
average_df["RH"] = average_predictions

plot_multiline_chart_pandas_using_index([train, test, average_df], "RH",
                                            ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
                                            "Time", "RH",
                                            "Relative Humidity Prediction Using Average",
                                            rotate_xticks=True)

print(20 * "-" + "NAIVE MODEL" + 20 * "-")
# %%------------------------------------------------------------------------------------------------------------
# naive model
naive_predictions = generic_naive_method(train["RH"], len(test["RH"]))
# print(naive_predictions)
naive_mse = cal_mse(test["RH"], naive_predictions)
print()
print("The MSE for Naive model is:")
print(naive_mse)

naive_rmse = np.sqrt(naive_mse)
print()
print("The RMSE for Naive model is:")
print(naive_rmse)

# forecast errors for naive model
residuals_naive = cal_forecast_errors(test["RH"], naive_predictions)

# naive residual variance
naive_variance = np.var(residuals_naive)
print()
print("The Variance of residual for Naive model is:")
print(naive_variance)

# naive residual mean
naive_mean = np.mean(residuals_naive)
print()
print("The Mean of residual for Naive model is:")
print(naive_mean)

# naive residual ACF
residual_autocorrelation_naive = cal_auto_correlation(residuals_naive, len(naive_predictions))
plot_acf(residual_autocorrelation_naive, "ACF plot using Naive Residuals")

# add the results to common dataframe
result_performance = result_performance.append(
pd.DataFrame(
            {"Model": ["Naive Model"], "MSE": [naive_mse], "RMSE": [naive_rmse],
             "Residual Mean": [naive_mean], "Residual Variance": [naive_variance]}))

# plot the predicted vs actual data
naive_df = test.copy(deep=True)
naive_df["RH"] = naive_predictions

plot_multiline_chart_pandas_using_index([train, test, naive_df], "RH",
                                            ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
                                            "Time", "RH",
                                            "Relative Humidity Prediction Using Naive Model",
                                            rotate_xticks=True)

print(20 * "-" + "DRIFT MODEL" + 20 * "-")
# %%------------------------------------------------------------------------------------------------------------
# drift model
drift_predictions = generic_drift_method(train["RH"], len(test["RH"]))
# print(drift_predictions)
drift_mse = cal_mse(test["RH"], drift_predictions)
print()
print("The MSE for drift model is:")
print(drift_mse)

drift_rmse = np.sqrt(drift_mse)
print()
print("The RMSE for Drift model is:")
print(drift_rmse)

# forecast errors for drift model
residuals_drift = cal_forecast_errors(test["RH"], drift_predictions)

# drift residual variance
drift_variance = np.var(residuals_drift)
print()
print("The Variance of residual for Drift model is:")
print(drift_variance)

# drift residual mean
drift_mean = np.mean(residuals_drift)
print()
print("The Mean of residual for drift model is:")
print(drift_mean)


# drift residual ACF
residual_autocorrelation_drift = cal_auto_correlation(residuals_drift, len(drift_predictions))
plot_acf(residual_autocorrelation_drift, "ACF plot using drift Residuals")


# add the results to common dataframe
result_performance = result_performance.append(
pd.DataFrame(
            {"Model": ["Drift Model"], "MSE": [drift_mse], "RMSE": [drift_rmse],
             "Residual Mean": [drift_mean], "Residual Variance": [drift_variance]}))

# plot the predicted vs actual data
drift_df = test.copy(deep=True)
drift_df["RH"] = drift_predictions

plot_multiline_chart_pandas_using_index([train, test, drift_df], "RH",
                                            ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
                                            "Time", "RH",
                                            "Relative Humidity Prediction Using Drift Model",
                                            rotate_xticks=True)


print(20 * "-" + "HOLT WINTER MODEL" + 20 * "-")
# %%------------------------------------------------------------------------------------------------------------

# holt winter prediction
holt_winter_prediction = generic_holt_linear_winter(train["RH"], test["RH"], None, None,
                                                        "mul", None)
# print(holt_winter_prediction)

# holt winter mse
holt_winter_mse = cal_mse(test["RH"], holt_winter_prediction)


print()
print("The MSE for Holt Winter model is:")
print(holt_winter_mse)

# holt winter rmse
holt_winter_rmse = np.sqrt(holt_winter_mse)
print()
print("The RMSE for Holt Winter model is:")
print(holt_winter_rmse)

# holt winter residual
residuals_holt_winter = cal_forecast_errors(list(test["RH"]), holt_winter_prediction)
residual_autocorrelation_holt_winter = cal_auto_correlation(residuals_holt_winter, len(holt_winter_prediction))

# holt winter residual variance
holt_winter_variance = np.var(residuals_holt_winter)
print()
print("The Variance of residual for Holt Winter model is:")
print(holt_winter_variance)

# holt winter residual mean
holt_winter_mean = np.mean(residuals_holt_winter)
print()
print("The Mean of residual for Holt Winter model is:")
print(holt_winter_mean)

# holt winter residual ACF
plot_acf(residual_autocorrelation_holt_winter, "ACF plot using Holt Winter Residuals")


# add the results to common dataframe
result_performance = result_performance.append(
pd.DataFrame(
            {"Model": ["Holt Winter Model"], "MSE": [holt_winter_mse], "RMSE": [holt_winter_rmse],
             "Residual Mean": [holt_winter_mean], "Residual Variance": [holt_winter_variance]}))

# plot the predicted vs actual data
holt_winter_df = test.copy(deep=True)
holt_winter_df["RH"] = holt_winter_prediction

plot_multiline_chart_pandas_using_index([train, test, holt_winter_df], "RH",
                                            ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
                                            "Time", "RH",
                                            "Relative Humidity Prediction Using Holt Winter",
                                            rotate_xticks=True)


print(20 * "-" + "SIMPLE AND EXPONENTIAL SMOOTHING" + 20 * "-")
# %%------------------------------------------------------------------------------------------------------------
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
y_hat_avg = test.copy()
fit2 = SimpleExpSmoothing(np.asarray(train["RH"])).fit(smoothing_level=0.6,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test))
plt.figure(figsize=(16,10))
plt.plot(train['RH'], label='Train')
plt.plot(test['RH'], label='Test')
plt.plot(y_hat_avg['SES'], label='SES')
plt.legend(loc='best')
plt.show()

# sse mse
sse_mse = cal_mse(test["RH"], y_hat_avg['SES'])


print()
print("The MSE for Holt Winter model is:")
print(sse_mse)

# sse rmse
sse_rmse = np.sqrt(sse_mse)
print()
print("The RMSE for Holt Winter model is:")
print(sse_rmse)

# sse residual
residuals_sse = cal_forecast_errors(list(test["RH"]), y_hat_avg['SES'])
residual_autocorrelation_sse = cal_auto_correlation(residuals_sse, len(y_hat_avg['SES']))

# sse residual variance
sse_variance = np.var(residuals_sse)
print()
print("The Variance of residual for Holt Winter model is:")
print(sse_variance)

# sse residual mean
sse_mean = np.mean(residuals_sse)
print()
print("The Mean of residual for Holt Winter model is:")
print(sse_mean)

# sse residual ACF
plot_acf(residual_autocorrelation_sse, "ACF plot using simple exponential smoothing Residuals")


# add the results to common dataframe
result_performance = result_performance.append(
pd.DataFrame(
            {"Model": [" Simple Exponential Smoothing Model"], "MSE": [sse_mse], "RMSE": [sse_rmse],
             "Residual Mean": [sse_mean], "Residual Variance": [sse_variance]}))


print(20 * "-" + "MULTIPLE LINEAR REGRESSION" + 20 * "-")
# %%------------------------------------------------------------------------------------------------------------
# combining train and test data
combined_data = train.append(test)

lm_combined = combined_data.copy(deep=True)
# separate train and test data
lm_train = lm_combined[:len(train)]
lm_test = lm_combined[len(train):]

# Scaling the data using MixMax Scaler
mm_scaler = MinMaxScaler()
lm_train_mm_scaled = pd.DataFrame(
mm_scaler.fit_transform(lm_train[np.setdiff1d(lm_train.columns, ["RH"])]),
        columns=np.setdiff1d(lm_train.columns, ["RH"]))
lm_train_mm_scaled.set_index(lm_train.index, inplace=True)
lm_train_mm_scaled["RH"] = lm_train["RH"]

lm_test_mm_scaled = pd.DataFrame(mm_scaler.transform(lm_test[np.setdiff1d(lm_test.columns, ["RH"])]),
                                     columns=np.setdiff1d(lm_test.columns, ["RH"]))
lm_test_mm_scaled.set_index(lm_test.index, inplace=True)
lm_test_mm_scaled["RH"] = lm_test["RH"]

print(lm_train_mm_scaled.columns)

# linear model using all variables
basic_model = normal_equation_using_statsmodels(
lm_train_mm_scaled[np.setdiff1d(lm_train_mm_scaled.columns, "RH")],
lm_train_mm_scaled["RH"], intercept=False)

print()
print("The summary of linear model with all variables is:")
print(basic_model.summary())


print(lm_train_mm_scaled.columns)
features_1 = np.setdiff1d(lm_train_mm_scaled.columns,
                            ["AH", "C6H6(GT)", "CO(GT)", "NMHC(GT)",
                             "NO2(GT)", "NOx(GT)", "PT08.S1(CO)", "PT08.S2(NMHC)",
                             "PT08.S3(NOx)", "PT08.S4(NO2)"])

pruned_model = normal_equation_using_statsmodels(lm_train_mm_scaled[np.setdiff1d(features_1, "RH")],
                                                     lm_train_mm_scaled["RH"], intercept=False)

print("The summary of linear model after feature selection:")
print(pruned_model.summary())

# linear model predictions
lm_predictions = normal_equation_prediction_using_statsmodels(pruned_model, lm_test_mm_scaled[
        np.setdiff1d(features_1, "RH")], intercept=False)

# linear model mse
lm_mse = cal_mse(test["RH"], lm_predictions)

print()
print("The MSE for Linear Model model is:")
print(lm_mse)

# linear model rmse
lm_rmse = np.sqrt(lm_mse)
print()
print("The RMSE for Linear Model model is:")
print(lm_rmse)

# plot the actual vs predicted values
lm_predictions_scaled = lm_test_mm_scaled.copy(deep=True)
lm_predictions_scaled["RH"] = lm_predictions


plot_multiline_chart_pandas_using_index([lm_train_mm_scaled, lm_test_mm_scaled, lm_predictions_scaled],
                                            "RH",
                                            ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
                                            "Time", "RH",
                                            "Relative Humidity Prediction Using Multiple Linear Model",
                                            rotate_xticks=True)

features = lm_train_mm_scaled.columns

# linear model predictions
lm_predictions = normal_equation_prediction_using_statsmodels(basic_model, lm_test_mm_scaled[
np.setdiff1d(features, "RH")], intercept=False)

# linear model mse
lm_mse = cal_mse(test["RH"], lm_predictions)

print()
print("The MSE for Linear Model model is:")
print(lm_mse)

# linear model rmse
lm_rmse = np.sqrt(lm_mse)
print()
print("The RMSE for Linear Model model is:")
print(lm_rmse)

# linear model residual
residuals_lm = cal_forecast_errors(list(test["RH"]), lm_predictions)
residual_autocorrelation = cal_auto_correlation(residuals_lm, len(lm_predictions))

# linear model residual variance
lm_variance = np.var(residuals_lm)
print()
print("The Variance of residual for Linear Model model is:")
print(lm_variance)

# linear model residual mean
lm_mean = np.mean(residuals_lm)
print()
print("The Mean of residual for Linear Model model is:")
print(lm_mean)

# linear model residual ACF
plot_acf(residual_autocorrelation, "ACF plot for Linear Model Residuals")

# linear model Q value
Q_value_lm = box_pierce_test(len(test), residuals_lm, len(test))
print()
print("The Q Value of residuals for Linear Model model is:")
print(Q_value_lm)

# add the results to common dataframe
result_performance = result_performance.append(
        pd.DataFrame(
            {"Model": ["Multiple Linear Regression Model"], "MSE": [lm_mse], "RMSE": [lm_rmse],
             "Residual Mean": [lm_mean], "Residual Variance": [lm_variance]}))

# plot the actual vs predicted values
lm_predictions_scaled = lm_test_mm_scaled.copy(deep=True)
lm_predictions_scaled["RH"] = lm_predictions

plot_multiline_chart_pandas_using_index([lm_train_mm_scaled, lm_test_mm_scaled, lm_predictions_scaled],
                                            "RH",
                                            ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
                                            "Time", "RH",
                                            "Relative Humidity Prediction Using Multiple Linear Model",
                                            rotate_xticks=True)

# Section 11: ARMA (ARIMA or SARIMA) model
# %%------------------------------------------------------------------------------------------------------------
print('# %%------------------------------------------------------------------------------------------------------------')
j = 12
k = 12
lags = j + k

y_mean = np.mean(train['RH'])
y = np.subtract(y_mean, df['RH'])
actual_output = np.subtract(y_mean, test['RH'])

# autocorrelation of RH
ry = auto_corr_cal(y, lags)

# create GPAC Table
gpac_table = create_gpac_table(j, k, ry)
print()
print("GPAC Table:")
print(gpac_table.to_string())
print()

plot_heatmap(gpac_table, "GPAC Table for RH")


# # estimate the order of the process
# # the possible orders identified from GPAC table don't pass the chi square test
possible_order2 = [(2, 3), (2, 7), (4, 0), (4, 2)]

print()
print("The possible orders identified from GPAC for ARMA process are:")
print(possible_order2)
print()
print("We noticed that none of the identified ARMA order from the GPAC table pass the chi squared test.")
print()

# checking which orders pass the GPAC test
print(gpac_order_chi_square_test(possible_order2, y, '2004-03-10 18:00:00', '2005-01-16 14:00:00',
                                  lags,actual_output))
# possible_order = [(2, 7)]
possible_order = [(2, 3)]
# possible_order = [(4, 0)]
# possible_order = [(4, 2)]

# checking which orders pass the GPAC test
gpac_order_chi_square_test(possible_order, y, '2004-03-10 18:00:00', '2005-01-16 14:00:00',
                                      lags, actual_output)

n_a = 2
n_b = 3

model = statsmodels_estimate_parameters(n_a, n_b, y)
print(model.summary())

# ARMA predictions
arma_prediction = statsmodels_predict_ARMA_process(model, "2005-01-16 15:00:00", "2005-04-04 14:00:00")

# # add the subtracted mean back into the predictions
arma_prediction = np.add(y_mean, arma_prediction)
# print(arma_prediction)

# def cal_mse(actual_values, predicted_values):
#     return np.round(np.mean(np.square(np.subtract(predicted_values, actual_values))), 3)

# ARMA mse
arma_mse = cal_mse(test["RH"], arma_prediction)

print(f"The MSE for ARMA({n_a}, {n_b}) model is:")
print(arma_mse)

# ARMA rmse
arma_rmse = np.sqrt(arma_mse)
print()

print(f"The RMSE for ARMA({n_a}, {n_b}) model is:")
print(arma_rmse)


# ARMA residual
residuals_arma = cal_forecast_errors(list(test["RH"]), arma_prediction)

# ARMA residual variance
arma_variance = np.var(residuals_arma)
print()
print("The Variance of residual for ARMA model is:")
print(arma_variance)

# ARMA residual mean
arma_mean = np.mean(residuals_arma)
print()
print(f"The Mean of residual for ARMA({n_a}, {n_b}) model is:")
print(arma_mean)

# ARMA residual ACF
residual_autocorrelation_arma = cal_auto_correlation(residuals_arma, len(arma_prediction))
plot_acf(residual_autocorrelation_arma, f"ACF plot for ARMA({n_a}, {n_b}) Residuals")

# ARMA covariance matrix
print()
print(statsmodels_print_covariance_matrix(model, n_a, n_b))

# ARMA estimated variance of error
statsmodels_print_variance_error(model, n_a, n_b)


# add the results to common dataframe
result_performance = result_performance.append(
        pd.DataFrame(
            {"Model": [f"ARMA({n_a}, {n_b}) Model"], "MSE": [arma_mse], "RMSE": [arma_rmse],
             "Residual Mean": [arma_mean], "Residual Variance": [arma_variance]}))

# plot the predicted vs actual data
arma_df = test.copy(deep=True)
arma_df["RH"] = arma_prediction

plot_multiline_chart_pandas_using_index([train, test, arma_df], "RH",
                                            ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
                                            "Time", "RH",
                                            f"Relative Humidity Prediction Using ARMA({n_a}, {n_b})",
                                            rotate_xticks=True)
# Section Extra: ARIMA (ARIMA or SARIMA) model
# %%------------------------------------------------------------------------------------------------------------

#Determine the rolling statistics
rolmean = df_rh.rolling(window=12).mean()
rolstd = df_rh.rolling(window=12).std()
# print(rolmean,rolstd)
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
rcParams['figure.figsize'] = 15,10
autocorrelation_plot(df_rh)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Time Series plot of Relative humidity')
plt.legend(loc='best')
plt.show()
pyplot.show()
rcParams['figure.figsize'] = 10,5
pyplot.plot(df_rh)
plt.xlabel('Time: March 2004- February 2005')
plt.ylabel('Relative Humidity (RH)')
plt.title('Time Series plot of Relative humidity')
plt.legend(loc='best')
plt.show()


#Estimate trend
df_RH_logScale = np.log(df_rh)
plt.plot(df_RH_logScale)
plt.xlabel('Time: March 2004- February 2005')
plt.ylabel('Relative Humidity (RH)')
plt.title('Time Series plot of Relative humidity in Log scale')
plt.legend(loc='best')
plt.show()

from statsmodels.tsa.stattools import adfuller
def test_stationary(timeseries):
    movingAverage = timeseries.rolling(window=12).mean()
    movingStd = timeseries.rolling(window=12).std()

    # Plot rolling
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(movingAverage, color='red', label='Rolling Average')
    std = plt.plot(movingStd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.xlabel('Time: March 2004- February 2005')
    plt.ylabel('Relative Humidity (RH)')
    plt.title('Time Series plot of Relative humidity')
    plt.legend(loc='best')
    plt.show()

    plt.show(block=False)

    # perform Dickey Fuller Test
    print('Result of Dickey Fuller Test')
    dftest = adfuller(timeseries['RH'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of observation Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)



exponentialDecayWeightedAverage = df_RH_logScale.ewm(halflife=12,min_periods=0,adjust=True).mean()
plt.plot(df_RH_logScale)
plt.plot(exponentialDecayWeightedAverage,color='red')
plt.xlabel('Time: March 2004- February 2005')
plt.ylabel('Relative Humidity (RH)')
plt.title('Time Series plot of Relative humidity in Log scale')
plt.legend(loc='best')
plt.show()


datasetLogScaleMinusMovingExponentialDecayAverage = df_RH_logScale - exponentialDecayWeightedAverage
test_stationary(datasetLogScaleMinusMovingExponentialDecayAverage)
datasetLogDiffShifting  = df_RH_logScale-df_RH_logScale.shift()
plt.plot(datasetLogDiffShifting)
plt.xlabel('Time: March 2004- February 2005')
plt.ylabel('Relative Humidity (RH)')
plt.title('Time Series plot of Relative humidity')
plt.legend(loc='best')
plt.show()
plt.show()

datasetLogDiffShifting.dropna(inplace=True)
test_stationary(datasetLogDiffShifting)

from statsmodels.tsa.arima_model import ARIMA
#ARIMMA Model
model = ARIMA(df_RH_logScale,order=(2, 1, 3))
# print(model)
result_ARIMA= model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(result_ARIMA.fittedvalues,color='red')
plt.title('RSS: %.4f'%sum((result_ARIMA.fittedvalues-datasetLogDiffShifting["RH"])**2))
print('Plotting ARIMA Model')
# plt.xlabel('Time: March 2004- February 2005')
# plt.ylabel('Relative Humidity (RH)')
# plt.title('Time Series plot of Relative humidity')
# plt.legend(loc='best')
plt.show()

prediction_ARIMA_diff = pd.Series(result_ARIMA.fittedvalues,copy=True)
prediction_ARIMA_diff_cumsum = prediction_ARIMA_diff.cumsum()
prediction_ARIMA_log = pd.Series(df_RH_logScale['RH'].iloc[0], index=df_RH_logScale.index)
prediction_ARIMA_log = prediction_ARIMA_log.add(prediction_ARIMA_diff_cumsum,fill_value =0)

prediction_ARIMA = np.exp(prediction_ARIMA_log)
plt.plot(df_rh)
plt.plot(prediction_ARIMA)
# plt.xlabel('Time: March 2004- February 2005')
# plt.ylabel('Relative Humidity (RH)')
# plt.title('ARIMA Prediction Plot')
# plt.legend(loc='best')
plt.show()



result_ARIMA.plot_predict(1,264)
plt.xlabel('Time: March 2004- February 2005')
plt.ylabel('Relative Humidity (RH)')
plt.title('Prediction v/s Forecast')
plt.legend(loc='best')
plt.show()



# print(result_ARIMA.summary().tables[1])

# ARMA mse
arima_mse = cal_mse(test["RH"], prediction_ARIMA)

print(f"The MSE for ARIMA  model is:")
print(arima_mse)

# ARMA rmse
arima_rmse = np.sqrt(arima_mse)
print()

print(f"The RMSE for ARIMA model is:")
print(arima_rmse)

# plot the predicted vs actual data
arima_df = test.copy(deep=True)
arima_df["RH"] = prediction_ARIMA

plot_multiline_chart_pandas_using_index([train, test, arima_df], "RH",
                                            ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
                                            "Time", "RH",
                                            f"Relative Humidity Prediction Using ARIMA",
                                            rotate_xticks=True)


# -------------------------------------------Final Performance Metrics----------------------
print()
print("The performance metrics for all the models is shown:")
print(result_performance.sort_values(["RMSE"]).reset_index(drop=True).to_string())
