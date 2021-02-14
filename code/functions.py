#Autocorrelation Function
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.tsa.holtwinters as ets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from pylab import rcParams
import statsmodels.api as sm
from scipy.stats import chi2
# %%------------------------------------------------------------------------------------------------------------
def split_df_train_test(df, test_size, random_seed=42):
    train, test = train_test_split(df, shuffle=False, test_size=test_size, random_state=random_seed)
    return train, test

# %%------------------------------------------------------------------------------------------------------------

def auto_corr(y,k):
    T = len(y)
    y_mean = np.mean(y)
    res_num = 0
    res_den = 0
    for t in range(k,T):
        res_num += (y[t] - y_mean) * (y[t-k] - y_mean)

    for t in range(0,T):
        res_den += (y[t] - y_mean)**2

    res = res_num/res_den
    return res

def auto_corr_cal(y,k):
    res = []
    for t in range(0,k):
        result = auto_corr(y,t)
        res.append(result)
    return res

def cal_auto_correlation(input_array, number_of_lags, precision=3):

    mean_of_input = np.mean(input_array)
    result = []
    denominator = np.sum(np.square(np.subtract(input_array, mean_of_input)))
    for k in range(0, number_of_lags):
        numerator = 0
        for i in range(k, len(input_array)):
            numerator += (input_array[i] - mean_of_input) * (input_array[i - k] - mean_of_input)
        if denominator != 0:
            result.append(np.round(numerator / denominator, precision))
    return result

def plot_acf(autocorrelation, title_of_plot, x_axis_label="Lags", y_axis_label="Magnitude"):
    # make a symmetric version of autocorrelation using slicing
    symmetric_autocorrelation = autocorrelation[:0:-1] + autocorrelation
    x_positional_values = [i * -1 for i in range(0, len(autocorrelation))][:0:-1] + [i for i in
                                                                                     range(0, len(autocorrelation))]
    # plot the symmetric version using stem
    rcParams['figure.figsize'] = 16, 10
    plt.stem(x_positional_values, symmetric_autocorrelation, use_line_collection=True)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title_of_plot)
    plt.figure(figsize=(16, 10))
    plt.show()


# %%------------------------------------------------------------------------------------------------------------
def adfuller_test(RH):
    result = adfuller(RH)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
# %%------------------------------------------------------------------------------------------------------------

def get_max_denominator_indices(j, k_scope):
    # create denominator indexes based on formula for GPAC
    denominator_indices = np.zeros(shape=(k_scope, k_scope), dtype=np.int64)

    for k in range(k_scope):
        denominator_indices[:, k] = np.arange(j - k, j + k_scope - k)

    return denominator_indices

def get_apt_denominator_indices(max_denominator_indices, k):
    apt_denominator_indices = max_denominator_indices[-k:, -k:]
    return apt_denominator_indices


def get_numerator_indices(apt_denominator_indices, k):
    numerator_indices = np.copy(apt_denominator_indices)
    # take the 0,0 indexed value and then create a range of values from (indexed_value+1, indexed_value+k)
    indexed_value = numerator_indices[0, 0]
    y_matrix = np.arange(indexed_value + 1, indexed_value + k + 1)

    # replace the last column with this new value
    numerator_indices[:, -1] = y_matrix

    return numerator_indices

def get_ACF_by_index(numpy_indices, acf):
    # select values from an array based on index specified
    result = np.take(acf, numpy_indices)
    return result

def get_phi_value(denominator_indices, numerator_indices, ry, precision=5):
    # take the absolute values since when computing phi value, we use ACF and ACF is symmetric in nature
    denominator_indices = np.abs(denominator_indices)
    numerator_indices = np.abs(numerator_indices)

    # replace the indices with the values of ACF
    denominator = get_ACF_by_index(denominator_indices, ry)
    numerator = get_ACF_by_index(numerator_indices, ry)

    # take the determinant
    denominator_det = np.round(np.linalg.det(denominator), precision)
    numerator_det = np.round(np.linalg.det(numerator), precision)

    # divide it and return the value of phi
    return np.round(np.divide(numerator_det, denominator_det), precision)

def create_gpac_table(j_scope, k_scope, ry, precision=5):
    # initialize gpac table
    gpac_table = np.zeros(shape=(j_scope, k_scope), dtype=np.float64)
    for j in range(j_scope):
        # create the largest denominator
        max_denominator_indices = get_max_denominator_indices(j, k_scope)

        for k in range(1, k_scope + 1):
            #  slicing largest denominator as required
            apt_denominator_indices = get_apt_denominator_indices(max_denominator_indices, k)

            # for numerator replace denominator's last columnn with index starting from j+1 upto k times
            numerator_indices = get_numerator_indices(apt_denominator_indices, k)

            # compute phi value
            phi_value = get_phi_value(apt_denominator_indices, numerator_indices, ry, precision)
            gpac_table[j, k - 1] = phi_value

    gpac_table_pd = pd.DataFrame(data=gpac_table, columns=[k for k in range(1, k_scope + 1)])

    return gpac_table_pd


def plot_heatmap(corr_df, title, xticks=None, yticks=None, x_axis_rotation=0, annotation=True):
    sns.heatmap(corr_df, annot=annotation)
    plt.title(title)
    if xticks is not None:
        plt.xticks([i for i in range(len(xticks))], xticks, rotation=x_axis_rotation)
    if yticks is not None:
        plt.yticks([i for i in range(len(yticks))], yticks)
    plt.show()
# %%------------------------------------------------------------------------------------------------------------
# average method
def generic_average_method(input_data, step_ahead):
    # returns a flat prediction
    return [np.round(np.mean(input_data), 3) for i in range(0, step_ahead)]

# naive method
def generic_naive_method(input_data, step_ahead):
    """Predicts using naive method for specified steps"""
    return [input_data[-1] for i in range(0, step_ahead)]

# drift method
def generic_drift_method(input_data, step_ahead):
    predicted_values = []

    for i in range(0, step_ahead):
        predicted_value = input_data[-1] + (i + 1) * ((input_data[-1] - input_data[0]) / (len(input_data) - 1))

        predicted_values.append(round(predicted_value, 3))

    return predicted_values

# holt_linear_winter_method
def generic_holt_linear_winter(train_data, test_data, seasonal_period: int, trend="mul", seasonal="mul",
                               trend_damped=False):
    """ Works best for data with trend and seasonality"""
    holt_winter = ets.ExponentialSmoothing(train_data, trend=trend, seasonal=seasonal,
                                           seasonal_periods=seasonal_period, damped=trend_damped).fit()
    holt_winter_forecast = list(holt_winter.forecast(len(test_data)))
    return holt_winter_forecast


# %%------------------------------------------------------------------------------------------------------------
# mean square error
def cal_mse(actual_values, predicted_values):
    return np.round(np.mean(np.square(np.subtract(predicted_values, actual_values))), 3)

# forecast errors is difference between  observed values and predicted values
def cal_forecast_errors(actual_values, predicted_values):
    # forecast errors is difference between  observed values and predicted values
    return np.subtract(actual_values, predicted_values)

# %%------------------------------------------------------------------------------------------------------------

def box_pierce_test(number_of_samples, residuals, lags):
    return round(number_of_samples * np.sum(np.square(cal_auto_correlation(residuals, lags + 1)[1:])), 3)

# %%------------------------------------------------------------------------------------------------------------
def plot_multiline_chart_pandas_using_index(list_of_dataframes, y_axis_common_data, list_of_label, list_of_color,
                                            x_label, y_label, title_of_plot, rotate_xticks=False):
    for i, df in enumerate(list_of_dataframes):
        df[y_axis_common_data].plot(label=list_of_label[i], color=list_of_color[i])
    plt.legend(loc='best')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title_of_plot)
    if rotate_xticks:
        plt.xticks(rotation=90)
    plt.figure(figsize=(16, 10))
    plt.show()
# %%------------------------------------------------------------------------------------------------------------
def plot_multiline_chart_pandas(list_of_dataframes, x_axis_common_data, y_axis_common_data, list_of_label,
                                list_of_color,
                                x_label, y_label, title_of_plot, rotate_xticks=False):
    """Plots multiple line charts into single chart. This API uses list of pandas data having same x_axis label and
    same y_axis label """
    for i, df in enumerate(list_of_dataframes):
        plt.plot(df[x_axis_common_data], df[y_axis_common_data], label=list_of_label[i], color=list_of_color[i])
    plt.legend(loc='best')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title_of_plot)
    if rotate_xticks:
        plt.xticks(rotation=90)
    plt.show()


def chi_square_test(Q, lags, n_a, n_b, alpha=0.01):
    dof = lags - n_a - n_b
    chi_critical = chi2.isf(alpha, df=dof)

    if Q < chi_critical:
        print(f"The residual is white and the estimated order is n_a= {n_a} and n_b = {n_b}")
    else:
        print(f"The residual is not white with n_a={n_a} and n_b={n_b}")

    return Q < chi_critical

def Q_value(y, autocorrelation_of_residuals):
    """ Computes Q value for comparing with chi_critical for Chi Square Test. Same as box_pierce_test(..)"""
    Q = len(y) * np.sum(np.square(autocorrelation_of_residuals[1:]))
    return Q

def statsmodels_estimate_parameters(n_a, n_b, y, trend="nc"):
    model = sm.tsa.ARMA(y, (n_a, n_b)).fit(trend=trend, disp=0)
    return model


def statsmodels_print_parameters(model, n_a, n_b):
    # print the parameters which are estimated
    for i in range(n_a):
        print("The AR coefficients a {}".format(i), "is:", model.params[i])
    print()
    for i in range(n_b):
        print("The MA coefficients b {}".format(i), "is:", model.params[i + n_a])
    print()


def statsmodels_print_covariance_matrix(model, n_a, n_b):
    print(f"Estimated covariance matrix for n_a = {n_a} and n_b = {n_b}: \n{model.cov_params()}")
    print()
    return model.cov_params()


def statsmodels_print_variance_error(model, n_a, n_b):
    print(f"Estimated variance of error for n_a = {n_a} and n_b = {n_b}: \n{model.sigma2}")
    print()
    return model.sigma2


def statsmodels_print_confidence_interval(model, n_a, n_b):
    # confidence interval
    print(
        f"The confidence interval for estimated parameters for n_a = {n_a} and n_b = {n_b}: \n {model.conf_int()}")
    print()
    return model.conf_int()


def statsmodels_predict_ARMA_process(model, start, stop):
    model_hat = model.predict(start=start, end=stop)
    return model_hat


def statsmodels_plot_predicted_true(y, model_hat, n_a, n_b):
    true_data = pd.DataFrame({"Magnitude": y, "Samples": [i for i in range(len(y))]})
    fitted_data = pd.DataFrame({"Magnitude": model_hat, "Samples": [i for i in range(len(model_hat))]})

    plot_multiline_chart_pandas([true_data, fitted_data], "Samples", "Magnitude", ["True data", "Fitted data"],
                                ["red", "blue"], "Samples", "Magnitude",
                                f"ARMA process with n_a={n_a} and n_b={n_b}")


def statsmodels_print_roots_AR(model):
    print("Real part:")
    for root in model.arroots:
        print(root.real)
    print("Imaginary part:")
    for root in model.arroots:
        print(root.imag)


def statsmodels_print_roots_MA(model):
    print("Real part:")
    for root in model.maroots:
        print(root.real)
    print("Imaginary part:")
    for root in model.maroots:
        print(root.imag)


def gpac_order_chi_square_test(possible_order_ARMA, train_data, start, stop, lags, actual_outputs):
    results = []

    for n_a, n_b in possible_order_ARMA:
        try:
            # estimate the model parameters
            model = statsmodels_estimate_parameters(n_a, n_b, train_data)


            # performing h step predictions
            predictions = statsmodels_predict_ARMA_process(model, start=start, stop=stop)

            # # add mean back to the forecast values
            # predictions = np.add(mean_to_add, predictions)

            # calculate forecast errors
            residuals = cal_forecast_errors(actual_outputs, predictions)

            # autocorrelation of residuals
            re = cal_auto_correlation(residuals, lags)

            # compute Q value for chi square test
            Q = Q_value(actual_outputs, re)

            # checking the chi square test
            if chi_square_test(Q, lags, n_a, n_b):
                results.append((n_a, n_b))

        except Exception as e:
            # print(e)
            pass

    return results

def normal_equation_using_statsmodels(train_feature_list, train_target_list, intercept=True):
    if intercept:
        train_feature_list = sm.add_constant(train_feature_list)

    model = sm.OLS(train_target_list, train_feature_list)
    results = model.fit()
    return results

def normal_equation_prediction_using_statsmodels(OLS_model, test_feature_list, intercept=True):
    if intercept:
        test_feature_list = sm.add_constant(test_feature_list, has_constant='add')

    predicted_values_OLS = OLS_model.predict(test_feature_list)
    return predicted_values_OLS



