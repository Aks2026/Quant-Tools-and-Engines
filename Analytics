import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import warnings
warnings.filterwarnings("ignore")
import math
import os
from datetime import date, timedelta, datetime
import time
from tqdm import tqdm
import pyodbc
from scipy import stats



def rolling_function(df, year):
    #greater than 1 year
    if year >= 1:
        df=df.set_index('Date').pct_change(250*year)
        # df=df.set_index('Date').pct_change(12*year)
        df=df+1
        df=df**(1/year)-1
        # print(df)
        if len(df.dropna()) == 0:
            df['Value'] = np.nan
            df['Nifty'] = np.nan
    else:
        #quaterly
        df=df.set_index('Date').pct_change(63)
    return df


def underwaterplot(draw, col):
    a = draw[col].cummax()
    b = a - draw[col]
    c = -b / a
    return c


def weekly_return_function(df):
    firstValue  = df.groupby(pd.Grouper(level='Date', freq='W')).nth(0).Value
    lastValue  = df.groupby(pd.Grouper(level='Date', freq='W')).nth(-1).Value
    # ret_s = (pd.DataFrame(lastValue/firstValue -1 )*100).reset_index()
    strategy_return = pd.DataFrame()
    strategy_return['Date'] = lastValue.reset_index()['Date']
    strategy_return['Value'] =  (lastValue.values/firstValue.values - 1)*100
    firstValue  = df.groupby(pd.Grouper(level='Date', freq='W')).nth(0).Nifty
    lastValue  = df.groupby(pd.Grouper(level='Date', freq='W')).nth(-1).Nifty
    # ret_b = (pd.DataFrame(lastValue/firstValue -1 )*100).reset_index()
    benchmark_return = pd.DataFrame()
    benchmark_return['Date'] = lastValue.reset_index()['Date']
    benchmark_return['Nifty'] =  (lastValue.values/firstValue.values - 1)*100
    calendar=pd.merge(strategy_return, benchmark_return)
    calendar = calendar.round(2)
    calendar["Date"] = calendar["Date"].astype(str)
    calendar.rename(columns={"Value": "Strategy", "Nifty": "Benchmark"}, inplace=True)
    return calendar


def monthly_return_function(df):
    firstValue  = df.groupby(pd.Grouper(level='Date', freq='M')).nth(0).Value
    lastValue  = df.groupby(pd.Grouper(level='Date', freq='M')).nth(-1).Value
    # ret_s = (pd.DataFrame(lastValue/firstValue -1 )*100).reset_index()
    strategy_return = pd.DataFrame()
    strategy_return['Date'] = lastValue.reset_index()['Date']
    strategy_return['Value'] =  (lastValue.values/firstValue.values - 1)*100
    firstValue  = df.groupby(pd.Grouper(level='Date', freq='M')).nth(0).Nifty
    lastValue  = df.groupby(pd.Grouper(level='Date', freq='M')).nth(-1).Nifty
    # ret_b = (pd.DataFrame(lastValue/firstValue -1 )*100).reset_index()
    benchmark_return = pd.DataFrame()
    benchmark_return['Date'] = lastValue.reset_index()['Date']
    benchmark_return['Nifty'] =  (lastValue.values/firstValue.values - 1)*100
    calendar=pd.merge(strategy_return, benchmark_return)
    calendar = calendar.round(2)
    calendar["Date"] = calendar["Date"].astype(str)
    calendar.rename(columns={"Value": "Strategy", "Nifty": "Benchmark"}, inplace=True)
    return calendar


def calendar_return_function(df):
    #strategy return
    firstValue  = df.groupby(pd.Grouper(level='Date', freq='A')).nth(0).Value
    lastValue  = df.groupby(pd.Grouper(level='Date', freq='A')).nth(-1).Value
    # ret_s = (pd.DataFrame(lastValue/firstValue -1 )*100).reset_index()
    strategy_return = pd.DataFrame()
    strategy_return['Date'] = lastValue.reset_index()['Date']
    strategy_return['Value'] =  (lastValue.values/firstValue.values - 1)*100
    #benchmark return
    firstValue  = df.groupby(pd.Grouper(level='Date', freq='A')).nth(0).Nifty
    lastValue  = df.groupby(pd.Grouper(level='Date', freq='A')).nth(-1).Nifty
    # ret_b = (pd.DataFrame(lastValue/firstValue -1 )*100).reset_index()
    benchmark_return = pd.DataFrame()
    benchmark_return['Date'] = lastValue.reset_index()['Date']
    benchmark_return['Nifty'] =  (lastValue.values/firstValue.values - 1)*100
    calendar=pd.merge(strategy_return, benchmark_return)
    calendar = calendar.round(2)
    calendar["Date"] = calendar["Date"].astype(str)
    calendar.rename(columns={"Value": "Strategy", "Nifty": "Benchmark"}, inplace=True)
    return calendar


def financial_year_return_function(df):
    # df['Year'] = df['Date'].dt.to_period('Q-MAR').dt.qyear
    firstValue  = df.groupby(pd.Grouper(level='Date', freq='A-MAR')).nth(0).Value
    lastValue  = df.groupby(pd.Grouper(level='Date', freq='A-MAR')).nth(-1).Value
    # ret_s = (pd.DataFrame(lastValue/firstValue -1 )*100).reset_index()
    strategy_return = pd.DataFrame()
    strategy_return['Date'] = lastValue.reset_index()['Date']
    strategy_return['Value'] =  (lastValue.values/firstValue.values - 1)*100
    firstValue  = df.groupby(pd.Grouper(level='Date', freq='A-MAR')).nth(0).Nifty
    lastValue  = df.groupby(pd.Grouper(level='Date', freq='A-MAR')).nth(-1).Nifty
    # ret_b = (pd.DataFrame(lastValue/firstValue -1 )*100).reset_index()
    benchmark_return = pd.DataFrame()
    benchmark_return['Date'] = lastValue.reset_index()['Date']
    benchmark_return['Nifty'] =  (lastValue.values/firstValue.values - 1)*100
    calendar=pd.merge(strategy_return, benchmark_return)
    calendar = calendar.round(2)
    calendar["Date"] = calendar["Date"].astype(str)
    calendar.rename(columns={"Value": "Strategy", "Nifty": "Benchmark"}, inplace=True)
    return calendar


def quaterly_return_function(df):
    firstValue  = df.groupby(pd.Grouper(level='Date', freq='Q')).nth(0).Value
    lastValue  = df.groupby(pd.Grouper(level='Date', freq='Q')).nth(-1).Value
    # ret_s = (pd.DataFrame(lastValue/firstValue -1 )*100).reset_index()
    strategy_return = pd.DataFrame()
    strategy_return['Date'] = lastValue.reset_index()['Date']
    strategy_return['Value'] =  (lastValue.values/firstValue.values - 1)*100
    firstValue  = df.groupby(pd.Grouper(level='Date', freq='Q')).nth(0).Nifty
    lastValue  = df.groupby(pd.Grouper(level='Date', freq='Q')).nth(-1).Nifty
    # ret_b = (pd.DataFrame(lastValue/firstValue -1 )*100).reset_index()
    benchmark_return = pd.DataFrame()
    benchmark_return['Date'] = lastValue.reset_index()['Date']
    benchmark_return['Nifty'] =  (lastValue.values/firstValue.values - 1)*100
    calendar=pd.merge(strategy_return, benchmark_return)
    calendar = calendar.round(2)
    calendar["Date"] = calendar["Date"].astype(str)
    calendar.rename(columns={"Value": "Strategy", "Nifty": "Benchmark"}, inplace=True)
    return calendar


def round_number(number):
    return round(number, 2)


def calculate_mean_best_worst(df):
    data_list = []
    data_list.append(round(df["Value"].mean() * 100, 2))
    data_list.append(round(df["Value"].max() * 100, 2))
    data_list.append(round(df["Value"].min() * 100, 2))
    data_list.append(round(df["Value"].median() * 100, 2))
    return data_list


def calculate_probability_of_values_in_range(df, lower_limit, upper_limit):
    df = df.dropna()
    if len(df) == 0:
        return np.nan
    df = df * 100
    total_values_in_range = len(df[(df["Value"] >= lower_limit) & (df["Value"] <= upper_limit)])
    total_values_in_df = len(df)
    percentage = (total_values_in_range / total_values_in_df) * 100
    return round(percentage, 2)


def calculate_probability_table(df, rolling_returns_data, row_index, confidence):
    data = []
    data.append(
        calculate_probability_of_values_in_range(
            df, rolling_returns_data[row_index][0] - confidence, rolling_returns_data[row_index][0] + confidence
        )
    )
    data.append(
        calculate_probability_of_values_in_range(
            df, rolling_returns_data[row_index][1] - (confidence * 2), rolling_returns_data[row_index][1]
        )
    )
    data.append(
        calculate_probability_of_values_in_range(
            df, rolling_returns_data[row_index][2], rolling_returns_data[row_index][2] + (confidence * 2)
        )
    )
    data.append(
        calculate_probability_of_values_in_range(
            df, rolling_returns_data[row_index][3] - confidence, rolling_returns_data[row_index][3] + confidence
        )
    )
    return data


def calculate_probability_range(df):
    data, low, high, increment_factor = [], -10, None, 10
    column = ['<-10', '-10 to 0', '0 to 10', '10 to 20', '20 to 30', '30 to 40', '40 to 50', '>50']
    for index in range(8):
        if index == 0:
            data.append(calculate_probability_of_values_in_range(df, float('-inf'), -10))
        elif index == 7:
            data.append(calculate_probability_of_values_in_range(df, 50, float('inf')))
        else:
            high = low + increment_factor
            data.append(calculate_probability_of_values_in_range(df, low, high))
            low = high
    return data, column


def calculate_cagr(df, column_name):
    data, value = [], None

    period_less_than_1_year = [2, 5, 21, 21*2, 21*3, 21*6]
    for current_period in period_less_than_1_year:
        value = np.nan
        if len(df) >= current_period:
            value = ((df[column_name].iloc[-1] / df[column_name].iloc[-current_period]) -1) * 100
        data.append(value)

    period_greater_than_1_year = [1, 2, 3, 5, 7, 10, 15]
    for current_period in period_greater_than_1_year:
        value = np.nan
        if len(df) >= 250*current_period:
            value = ((df[column_name].iloc[-1] / df[column_name].iloc[-250*current_period]) ** (1/current_period) -1) * 100
        data.append(value)

    #cagr SI
    value = np.nan
    if len(df) >= 250:
        value = (((df[column_name].iloc[-1] / df[column_name].iloc[0]) ** (1/(len(df)/250))) -1) * 100
    elif len(df) > 0:
        value = ((df[column_name].iloc[-1] / df[column_name].iloc[0]) -1) * 100
    data.append(value)

    period_greater_than_1_year = [1, 2, 3, 5, 7, 10, 15]
    for current_period in period_greater_than_1_year:
        value = np.nan
        if len(df) >= 250*current_period:
            value = (df[column_name].pct_change(250*current_period)+1).iloc[-1]
        data.append(value)

    #absolute return SI
    value = np.nan
    if len(df) > 0:
        value = ((df[column_name].iloc[-1] / df[column_name].iloc[0]) -1)
    data.append(value)
    
    data = list(map(round_number, data))

    columns = ["1 Day", "1 Week", "1 Month", "2 Month", "3 Month", "6 Month", "CAGR 1", "CAGR 2", "CAGR 3", "CAGR 5", "CAGR 7", 
               "CAGR 10", "CAGR 15", "CAGR SI", "xtimes 1", "xtimes 2", "xtimes 3", "xtimes 5", "xtimes 7", "xtimes 10", "xtimes 15", 
               "xtimes SI"]
    
    return data, columns


def drawdown(draw, column_name):
    a=draw[column_name].cummax()
    b=a-draw[column_name]
    c=b/a
    d=(c.max())*-100
    return d


def calculate_beta(df):
    df["Daily Return Strategy"] = df["Value"].pct_change()
    df["Daily Return Benchmark"] = df["Nifty"].pct_change()
    # cov = df[{"Daily Return Strategy", "Daily Return Benchmark"}].cov()
    cov = df[["Daily Return Strategy", "Daily Return Benchmark"]].cov()
    var = df["Daily Return Benchmark"].var()
    beta = cov.loc["Daily Return Strategy", "Daily Return Benchmark"] / var
    return beta


def ratios1(df, column_name, beta, return_table, one_year_rolling_cagr):
    data, rfr = [], 0.07
    df['Daily Return'] = df[column_name].pct_change()
    df['Benchmark Return'] = df['Nifty'].pct_change()
    df['alpha_return'] = df['Daily Return'] - df['Benchmark Return']
    strategy_return = df['Daily Return'].mean()*252
    benchmark_return = df['Benchmark Return'].mean()*252
    std_of_alpha = (252 ** 0.5)*df['alpha_return'].std(ddof=0)

    #beta
    data.append(calculate_beta(df))

    #sharpe
    a=(df['Daily Return'].mean()*252)-rfr
    b=(252 ** 0.5)*df['Daily Return'].std(ddof=0)
    data.append(a/b)

    #sortino
    c=(252 ** 0.5)*df[df['Daily Return']<0][['Daily Return']].std(ddof=0).iloc[0]
    data.append(a/c)

    #max drawdown
    data.append(-1*drawdown(df, column_name))

    #treynor ratio
    beta = calculate_beta(df)
    if column_name == "Value":
        cagr_si = return_table["Strategy"]["CAGR SI"]
    else:
        cagr_si = return_table["Benchmark"]["CAGR SI"]
    data.append((cagr_si-6)/(100*beta))

    #jensen alpha
    data.append((cagr_si-(6+beta*(return_table["Benchmark"]["CAGR SI"]-rfr*100))))

    #std
    data.append(df['Daily Return'].std()*(252**(1/2))*100)

    #information ratio
    information_ratio = (strategy_return - benchmark_return)/std_of_alpha
    data.append(information_ratio)

    #upside capture ratio
    if column_name == 'Value':
        value_mean = one_year_rolling_cagr[one_year_rolling_cagr['Nifty'] >= 0]['Value'].mean()
        nifty_mean = one_year_rolling_cagr[one_year_rolling_cagr['Nifty'] >= 0]['Nifty'].mean()
        upside_capture_ratio = round(value_mean/nifty_mean, 2)
    else:
        upside_capture_ratio = np.nan
    data.append(upside_capture_ratio)

    #downside capture ratio
    if column_name == 'Value':
        value_mean = one_year_rolling_cagr[one_year_rolling_cagr['Nifty'] < 0]['Value'].mean()
        nifty_mean = one_year_rolling_cagr[one_year_rolling_cagr['Nifty'] < 0]['Nifty'].mean()
        downside_capture_ratio = round(value_mean/nifty_mean, 2)
    else:
        downside_capture_ratio = np.nan
    data.append(downside_capture_ratio)

    data = list(map(round_number, data))

    return data


def calculate_win_rate_and_loss_rate(df):
    total_benchmark_positive = np.count_nonzero(np.where((df["Benchmark"] > 0), 1, 0))
    total_benchmark_negative = np.count_nonzero(np.where((df["Benchmark"] < 0), 1, 0))
    positive_win_rate = np.count_nonzero(np.where((df["Benchmark"] > 0) & (df["Strategy"] > df["Benchmark"]), 1, 0))
    negative_win_rate = np.count_nonzero(np.where((df["Benchmark"] < 0) & (df["Strategy"] > df["Benchmark"]), 1, 0))
    a = np.nan
    if total_benchmark_positive != 0:
        a = round(positive_win_rate/total_benchmark_positive*100, 2)
    b = np.nan
    if total_benchmark_negative != 0:
        b = round(negative_win_rate/total_benchmark_negative*100, 2)
    return a, b


def calculate_rolling_sharpe(df, year, daily_log_with_nifty):
    rfr = 0.07
    df = df['Value'][-252*year:].mean()
    a = df - rfr
    b = daily_log_with_nifty["Value"].pct_change().rolling(window=252*year).std()*(252**0.5)
    return a/b


def calculate_rolling_std(year, daily_log_with_nifty):
    return daily_log_with_nifty["Value"].pct_change().rolling(window=252*year).std()*(252**0.5)


def calculate_rolling_sortino(df, year, daily_log_with_nifty):
    rfr = 0.07
    df = df['Value'][-252*year:].mean()
    daily_log_with_nifty['pct_change'] = daily_log_with_nifty['Value'].pct_change()
    a = df - rfr
    b = daily_log_with_nifty[daily_log_with_nifty['pct_change'] < 0]['pct_change'].std()*(252**0.5)
    return round(a/b, 2)


def monthly_quaterly_calendar_return(df, name):
    strategy, benchmark= [], []

    df = df.set_index("Date")

    #total positive count
    strategy.append(len(df[df["Strategy"] > 0]["Strategy"]))
    benchmark.append(len(df[df["Benchmark"] > 0]["Benchmark"]))

    #positive year return percentage
    strategy.append(len(df[df["Strategy"] > 0]["Strategy"]) / len(df["Strategy"]) * 100)
    benchmark.append(len(df[df["Benchmark"] > 0]["Benchmark"]) / len(df["Benchmark"]) * 100)

    #average absolute return mean of all
    strategy.append(df["Strategy"].mean())
    benchmark.append(df["Benchmark"].mean())

    #mean of only positives
    strategy.append(df[df["Strategy"] > 0]["Strategy"].mean())
    benchmark.append(df[df["Benchmark"] > 0]["Benchmark"].mean())

    #mean of only negatives
    strategy.append(df[df["Strategy"] < 0]["Strategy"].mean())
    benchmark.append(df[df["Benchmark"] < 0]["Benchmark"].mean())

    #max return
    strategy.append(df["Strategy"].max())
    benchmark.append(df["Benchmark"].max())

    #min return
    strategy.append(df["Strategy"].min())
    benchmark.append(df["Benchmark"].min())

    #std of all
    strategy.append(df["Strategy"].std())
    benchmark.append(df["Benchmark"].std())

    #round off to 2 digits
    strategy = list(map(round_number, strategy))
    benchmark = list(map(round_number, benchmark))

    absolute_stats = pd.DataFrame(columns=[f"Strategy {name}", f"Benchmark {name}"], 
                index=["Total Positive Count", f"Positive {name} Percentage", "Average Absolute Return", "Mean Of Positives", "Mean Of Negatives", "Max Return", "Min Return", "Std"])
    
    absolute_stats[f"Strategy {name}"] = strategy
    absolute_stats[f"Benchmark {name}"] = benchmark

    alpha_df = pd.DataFrame(df["Strategy"] - df["Benchmark"], columns=[f"Alpha {name}"])
    
    return absolute_stats, alpha_df


def calculate_alpha_stats(df, name):

    alpha_data = []

    #total positive count
    alpha_data.append(len(df[df[name[:-6]] > 0][name[:-6]]))

    #positive year return percentage
    alpha_data.append(len(df[df[name[:-6]] > 0][name[:-6]]) / len(df[name[:-6]]) * 100)

    #average absolute return mean of all
    alpha_data.append(df[name[:-6]].mean())

    #mean of only positives
    alpha_data.append(df[df[name[:-6]] > 0][name[:-6]].mean())

    #mean of only negatives
    alpha_data.append(df[df[name[:-6]] < 0][name[:-6]].mean())

    #max return
    alpha_data.append(df[name[:-6]].max())

    #min return
    alpha_data.append(df[name[:-6]].min())

    #std of all
    alpha_data.append(df[name[:-6]].std())

    alpha_stats = pd.DataFrame(alpha_data, columns=[f"{name}"], 
                index=["Total Positive Count", f"Positive {name} Percentage", "Average Absolute Return", "Mean Of Positives", "Mean Of Negatives", "Max Return", "Min Return", "Std"])
    

    return alpha_stats


def calculate_consecutive_wins_and_loss(df, win):
    count, max_count = 0, 0
    if win == True:
        #consecutive wins
        for index in range(len(df)):
            if df.iloc[index, 1] > df.iloc[index, 2]:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
    else:
        #consecutive loss
        for index in range(len(df)):
            if df.iloc[index, 1] < df.iloc[index, 2]:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
    return max_count


def to_drawdown_series(returns):
    """Convert returns series to drawdown series"""
    prices = returns
    dd = prices / np.maximum.accumulate(prices) - 1.
    return dd.replace([np.inf, -np.inf, -0], 0)


def drawdown_details(drawdown):
    """
    Calculates drawdown details, including start/end/valley dates,
    duration, max drawdown and max dd for 99% of the dd period
    for every drawdown period
    """
    def _drawdown_details(drawdown):
        # mark no drawdown
        no_dd = drawdown == 0

        # extract dd start dates
        starts = ~no_dd & no_dd.shift(1)
        starts = list(starts[starts].index)

        # extract end dates
        ends = no_dd & (~no_dd).shift(1)
        ends = list(ends[ends].index)

        # no drawdown :)
        if not starts:
            return pd.DataFrame(
                index=[], columns=('start', 'valley', 'end', 'days',
                                   'max drawdown', '99% max drawdown'))

        # drawdown series begins in a drawdown
        if ends and starts[0] > ends[0]:
            starts.insert(0, drawdown.index[0])

        # series ends in a drawdown fill with last date
        if not ends or starts[-1] > ends[-1]:
            ends.append(drawdown.index[-1])

        # build dataframe from results
        data = []
        for i, _ in enumerate(starts):
            dd = drawdown[starts[i]:ends[i]]
            clean_dd = dd
            data.append((starts[i], dd.idxmin(), ends[i],
                         (ends[i] - starts[i]).days,
                         dd.min() * 100, clean_dd.min() * 100))

        df = pd.DataFrame(data=data,
                           columns=('start', 'valley', 'end', 'days',
                                    'max drawdown',
                                    '99% max drawdown'))
        df['days'] = df['days'].astype(int)
        df['max drawdown'] = df['max drawdown'].astype(float)
        df['99% max drawdown'] = df['99% max drawdown'].astype(float)

        df['start'] = df['start'].dt.strftime('%Y-%m-%d')
        df['end'] = df['end'].dt.strftime('%Y-%m-%d')
        df['valley'] = df['valley'].dt.strftime('%Y-%m-%d')

        return df

    if isinstance(drawdown,pd.DataFrame):
        _dfs = {}
        for col in drawdown.columns:
            _dfs[col] = _drawdown_details(drawdown[col])
        return pd.concat(_dfs, axis=1)

    return drawdown_details(drawdown)


def days_to_recover(returns, benchmark=None, rf=0., grayscale=False,
         figsize=(8, 5), display=True, compounded=True,
         periods_per_year=252, match_dates=False):

    dd = to_drawdown_series(returns)
    col = drawdown_details(dd).columns[4]
    dd_info = drawdown_details(dd).sort_values(by = col, ascending = True)[:5]

    if not dd_info.empty:
        dd_info.index = range(1, min(6, len(dd_info)+1))
        dd_info.columns = map(lambda x: str(x).title(), dd_info.columns)

    return dd_info


def drawdown_function(df):
    days_to_recover_dataframe = days_to_recover(df, rf=0.07)
    days_to_recover_dataframe.columns=('Peak', 'Trough', 'Recover', 'Recovery days', 'Max drawdown', '99% max drawdown')
    days_to_recover_dataframe = days_to_recover_dataframe[days_to_recover_dataframe.columns]
    days_to_recover_dataframe['Trough'] = pd.to_datetime(days_to_recover_dataframe['Trough'])
    days_to_recover_dataframe['Recover'] = pd.to_datetime(days_to_recover_dataframe['Recover'])
    days_to_recover_dataframe['Peak'] = pd.to_datetime(days_to_recover_dataframe['Peak'])
    days_to_recover_dataframe['Peak To Trough Days'] = days_to_recover_dataframe['Trough'] - days_to_recover_dataframe['Peak']
    days_to_recover_dataframe['Recovery days'] = days_to_recover_dataframe['Recover'] - days_to_recover_dataframe['Trough']
    days_to_recover_dataframe = days_to_recover_dataframe.drop('99% max drawdown', axis=1)
    days_to_recover_dataframe['Peak'] = days_to_recover_dataframe['Peak'].astype(str)
    days_to_recover_dataframe['Trough'] = days_to_recover_dataframe['Trough'].astype(str)
    days_to_recover_dataframe['Recover'] = days_to_recover_dataframe['Recover'].astype(str)
    days_to_recover_dataframe = days_to_recover_dataframe[['Peak', 'Trough', 'Max drawdown', 'Peak To Trough Days', 'Recover', 'Recovery days']]
    days_to_recover_dataframe.rename(columns={'Max drawdown': 'Max Drawdown', 'Recover': 'Recovery Date', 'Recovery days': 'Recovery Days'}, inplace=True)
    days_to_recover_dataframe['Max Drawdown'] = round(days_to_recover_dataframe['Max Drawdown'], 2)
    return days_to_recover_dataframe


def calculate_geometric_mean(df):
    if len(df) == 0:
        return np.nan
    return (np.prod(df+1)**(1/len(df)))-1


# def calculate_average_days_to_recover(df, column_index):
#     total = 0
#     for index in range(len(df)):
#         total += df.iloc[index, column_index]
#     return total / len(df)


def cagr_si(df, column_name):
    value = (((df[column_name].iloc[-1] / df[column_name].iloc[0]) ** (1/(len(df)/250))) -1) * 100
    return round(value, 2)


def main_function(input_csv_file):

    # read file
    daily_log_with_nifty = pd.read_csv(input_csv_file)
    daily_log_with_nifty["Date"] = pd.to_datetime(daily_log_with_nifty["Date"])
    # daily_log_with_nifty["Date"] = pd.to_datetime(daily_log_with_nifty["Date"], format='%d-%m-%Y')
    #remove 0 values
    # daily_log_with_nifty = daily_log_with_nifty[daily_log_with_nifty["Value"] > 10]
    #filter till 1st Jan 2021
    # daily_log_with_nifty = daily_log_with_nifty[daily_log_with_nifty["Date"] >= '2021-01-01'].reset_index(drop=True)
    # print(input_csv_file)
    # daily_log_with_nifty["Value"] = (daily_log_with_nifty["Value"] / daily_log_with_nifty.loc[0, 'Value']) * 100
    # daily_log_with_nifty["Nifty"] = (daily_log_with_nifty["Nifty"] / daily_log_with_nifty.loc[0, 'Nifty']) * 100
    # daily_log_with_nifty = daily_log_with_nifty[['Date', 'Value', 'Nifty']]
    # print(daily_log_with_nifty)
    

    #display
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 2)
     

    # rolling cagr
    quaterly_rolling_cagr = rolling_function(daily_log_with_nifty, 0.25)
    one_year_rolling_cagr = rolling_function(daily_log_with_nifty, 1)
    three_year_rolling_cagr = rolling_function(daily_log_with_nifty, 3)
    five_year_rolling_cagr = rolling_function(daily_log_with_nifty, 5)
    seven_year_rolling_cagr = rolling_function(daily_log_with_nifty, 7)
    ten_year_rolling_cagr = rolling_function(daily_log_with_nifty, 10)


    # yearly, quaterly and monthly return
    calendar_return = calendar_return_function(daily_log_with_nifty.set_index("Date"))
    monthly_return = monthly_return_function(daily_log_with_nifty.set_index("Date"))
    quaterly_return = quaterly_return_function(daily_log_with_nifty.set_index("Date"))

    #closing_nav
    closing_nav = round(daily_log_with_nifty['Value'][len(daily_log_with_nifty)-1], 2)

    #CYTD
    CYTD = calendar_return['Strategy'][len(calendar_return)-1]

    #FYTD
    latest_month = daily_log_with_nifty['Date'].dt.month.iloc[-1]
    latest_year = daily_log_with_nifty['Date'].dt.year.to_numpy()[-1]
    if latest_month < 4: 
        #for date 2024-01-01 month is jan so FYTD will be from 2023-04-01 to 2024-03-31 so latest year is 2023
        latest_year -= 1
    ls = daily_log_with_nifty[daily_log_with_nifty['Date'] >= f'{latest_year}-04-01']['Value'].to_numpy()
    if len(ls) >= 2:
        first_value = ls[0]
        last_value = ls[-1]
        FYTD = round((last_value/first_value - 1)*100, 2)
    else:
        FYTD = np.nan

    
    # drawdown
    strategy = underwaterplot(daily_log_with_nifty, "Value")
    benchmark = underwaterplot(daily_log_with_nifty, "Nifty")


    # return tables
    rolling_returns_data = []
    rolling_returns_data.append(calculate_mean_best_worst(one_year_rolling_cagr))
    rolling_returns_data.append(calculate_mean_best_worst(three_year_rolling_cagr))
    rolling_returns_data.append(calculate_mean_best_worst(five_year_rolling_cagr))

    rolling_cagr_probability_data = []
    rolling_cagr_probability_data.append(calculate_probability_table(three_year_rolling_cagr, rolling_returns_data, 1, 2))
    rolling_cagr_probability_data.append(calculate_probability_table(five_year_rolling_cagr, rolling_returns_data, 2, 2))
    rolling_returns_data.insert(2, rolling_cagr_probability_data[0])
    rolling_returns_data.append(rolling_cagr_probability_data[1])

    rolling_returns = pd.DataFrame(
        rolling_returns_data,
        columns=["Mean", "Best", "Worst", "Median"],
        index=[
            "1 Year Rolling CAGR",
            "3 Year Rolling CAGR",
            "3 Year Rolling CAGR probability",
            "5 Year Rolling CAGR",
            "5 Year Rolling CAGR probability",
        ],
    )

    # print(rolling_returns)

    three_year_rolling_cagr_probability_range_data, three_year_column_name = calculate_probability_range(three_year_rolling_cagr)
    five_year_rolling_cagr_probability_range_data, five_year_column_name = calculate_probability_range(five_year_rolling_cagr)

    three_year_rolling_cagr_probability_chart = pd.DataFrame(columns=three_year_column_name, index=["3 Year Rolling CAGR"])
    three_year_rolling_cagr_probability_chart.loc["3 Year Rolling CAGR"] = three_year_rolling_cagr_probability_range_data

    five_year_rolling_cagr_probability_chart = pd.DataFrame(columns=five_year_column_name, index=["5 Year Rolling CAGR"])
    five_year_rolling_cagr_probability_chart.loc["5 Year Rolling CAGR"] = five_year_rolling_cagr_probability_range_data

    strategy, column_name = calculate_cagr(daily_log_with_nifty, "Value")
    benchmark, column_name = calculate_cagr(daily_log_with_nifty, "Nifty")

    return_table = pd.DataFrame(columns=["Strategy", "Benchmark"], index=column_name)

    return_table["Strategy"] = strategy
    return_table["Benchmark"] = benchmark

    # print(return_table)

    
    # risk ratios
    beta = calculate_beta(daily_log_with_nifty)
    strategy = ratios1(daily_log_with_nifty, "Value", beta, return_table, one_year_rolling_cagr)
    benchmark = ratios1(daily_log_with_nifty, "Nifty", beta, return_table, one_year_rolling_cagr)

    risk_ratios = pd.DataFrame(columns=["Strategy", "Benchmark"], 
                           index=["Beta", "Sharpe", "Sortino", "Max Drawdown", "Treynor Ratio", "Jensen Alpha", "STD", "Information Ratio", 
                                  "Upside Capture Ratio", "Downside Capture Ratio"])
    risk_ratios["Strategy"] = strategy
    risk_ratios["Benchmark"] = benchmark
    # print(risk_ratios)

    #r square
    correlation_matrix = np.corrcoef(daily_log_with_nifty['Value'], daily_log_with_nifty['Nifty'])
    coefficient_of_correlation = correlation_matrix[0, 1]
    # r = coefficient_of_correlation
    r_square = round(coefficient_of_correlation**2, 2)
    
    # alpha percentage, positive win rate and negative win rate
    alpha_percentage_data = []
    calendar_alpha = (len(calendar_return[calendar_return["Strategy"] > calendar_return["Benchmark"]])/len(calendar_return))*100
    alpha_percentage_data.append(round(calendar_alpha, 2))
    quaterly_alpha = (len(quaterly_return[quaterly_return["Strategy"] > quaterly_return["Benchmark"]])/len(quaterly_return))*100
    alpha_percentage_data.append(round(quaterly_alpha, 2))
    monthly_alpha =  (len(monthly_return[monthly_return["Strategy"] > monthly_return["Benchmark"]])/len(monthly_return))*100
    alpha_percentage_data.append(round(monthly_alpha, 2))

    positive_win_rate_data, negative_win_rate_data = [], []
    positive_win_rate, negative_win_rate = calculate_win_rate_and_loss_rate(calendar_return)
    positive_win_rate_data.append(positive_win_rate)
    negative_win_rate_data.append(negative_win_rate)
    positive_win_rate, negative_win_rate = calculate_win_rate_and_loss_rate(quaterly_return)
    positive_win_rate_data.append(positive_win_rate)
    negative_win_rate_data.append(negative_win_rate)
    positive_win_rate, negative_win_rate = calculate_win_rate_and_loss_rate(monthly_return)
    positive_win_rate_data.append(positive_win_rate)
    negative_win_rate_data.append(negative_win_rate)

    absolute_positive_data = []
    absolute_positive_data.append(round(len(calendar_return[calendar_return['Strategy'] > 0]) / len(calendar_return) * 100, 2))
    absolute_positive_data.append(round(len(quaterly_return[quaterly_return['Strategy'] > 0]) / len(quaterly_return) * 100, 2))
    absolute_positive_data.append(round(len(monthly_return[monthly_return['Strategy'] > 0]) / len(monthly_return) * 100, 2))

    alpha_percentage = pd.DataFrame(index=["Calendar", "Quaterly", "Monthly"])
    alpha_percentage["Alpha Ratio"] = alpha_percentage_data
    alpha_percentage["Positive Win Rate"] = positive_win_rate_data
    alpha_percentage["Negative Win Rate"] = negative_win_rate_data
    alpha_percentage["Absolute Positive"] = absolute_positive_data
    # print(alpha_percentage)



    # heatmaps
    quaterly_return_function_date = quaterly_return_function(daily_log_with_nifty.set_index("Date"))
    quaterly_return_function_date["Date"] = pd.to_datetime(quaterly_return_function_date["Date"])
    quaterly_heatmap_data = quaterly_return_function_date
    quaterly_heatmap_data["Year"] = quaterly_heatmap_data["Date"].dt.year
    quaterly_heatmap_data["Quarter"] = quaterly_heatmap_data["Date"].dt.month
    # quaterly_heatmap_data = (quaterly_heatmap_data[{"Year", "Quarter", "Strategy"}].groupby(["Year", "Quarter"]).sum().unstack("Quarter"))
    quaterly_heatmap_data = (quaterly_heatmap_data[["Year", "Quarter", "Strategy"]].groupby(["Year", "Quarter"]).sum().unstack("Quarter"))
    quaterly_heatmap_data = round(quaterly_heatmap_data, 2)
    # a4_dims = (12, 6)
    # fig, ax = plt.subplots(figsize=a4_dims)
    # sns.heatmap(quaterly_heatmap_data, annot=True, cmap="RdYlGn").set(title="Quaterly Absolute Heatmap")
    # sns.heatmap(calendar, annot=True,cmap="RdYlGn")

    quaterly_heatmap_data_alpha = quaterly_return_function_date
    quaterly_alpha = quaterly_return_function_date["Strategy"] - quaterly_return_function_date["Benchmark"]
    quaterly_heatmap_data_alpha["Year"] = quaterly_heatmap_data_alpha["Date"].dt.year
    quaterly_heatmap_data_alpha["Quarter"] = quaterly_heatmap_data_alpha["Date"].dt.month
    quaterly_heatmap_data_alpha["Alpha"] = quaterly_alpha
    # quaterly_heatmap_data_alpha = (quaterly_heatmap_data_alpha[{"Year", "Quarter", "Alpha"}].groupby(["Year", "Quarter"]).sum().unstack("Quarter"))
    quaterly_heatmap_data_alpha = (quaterly_heatmap_data_alpha[["Year", "Quarter", "Alpha"]].groupby(["Year", "Quarter"]).sum().unstack("Quarter"))
    quaterly_heatmap_data_alpha = round(quaterly_heatmap_data_alpha, 2)
    # a4_dims = (12, 6)
    # fig, ax = plt.subplots(figsize=a4_dims)
    # sns.heatmap(quaterly_heatmap_data_alpha, annot=True, cmap="RdYlGn").set(title="Quaterly Alpha Heatmap")
    # sns.heatmap(calendar, annot=True,cmap="RdYlGn")

    monthly_return_function_date = monthly_return_function(daily_log_with_nifty.set_index("Date"))
    monthly_return_function_date["Date"] = pd.to_datetime(monthly_return_function_date["Date"])
    monthly_heatmap_data = monthly_return_function_date
    monthly_heatmap_data["Year"] = monthly_heatmap_data["Date"].dt.year
    monthly_heatmap_data["Monthly"] = monthly_heatmap_data["Date"].dt.month
    monthly_heatmap_data = (monthly_heatmap_data[["Year", "Monthly", "Strategy"]].groupby(["Year", "Monthly"]).sum().unstack("Monthly"))
    monthly_heatmap_data = round(monthly_heatmap_data, 2)
    # a4_dims = (12, 6)
    # fig, ax = plt.subplots(figsize=a4_dims)
    # sns.heatmap(monthly_heatmap_data, annot=True, cmap="RdYlGn").set(title="Monthly Absolute Heatmap")
    # sns.heatmap(calendar, annot=True,cmap="RdYlGn")

    monthly_heatmap_data_alpha = monthly_return_function_date
    monthly_alpha = monthly_return_function_date["Strategy"] - monthly_return_function_date["Benchmark"]
    monthly_heatmap_data_alpha["Year"] = monthly_heatmap_data_alpha["Date"].dt.year
    monthly_heatmap_data_alpha["Monthly"] = monthly_heatmap_data_alpha["Date"].dt.month
    monthly_heatmap_data_alpha["Alpha"] = monthly_alpha
    monthly_heatmap_data_alpha = (monthly_heatmap_data_alpha[["Year", "Monthly", "Alpha"]].groupby(["Year", "Monthly"]).sum().unstack("Monthly"))
    monthly_heatmap_data_alpha = round(monthly_heatmap_data_alpha, 2)
    # a4_dims = (12, 6)
    # fig, ax = plt.subplots(figsize=a4_dims)
    # sns.heatmap(monthly_heatmap_data_alpha, annot=True, cmap="RdYlGn").set(title="Monthly Alpha Heatmap")
    # sns.heatmap(calendar, annot=True,cmap="RdYlGn")

    # comment below 3 lines after first time run
    calendar_return["Date"] = pd.to_datetime(calendar_return["Date"])
    calendar_return["Date"] = calendar_return["Date"].dt.year
    calendar_return = calendar_return.set_index("Date")
    # a4_dims = (12, 6)
    # fig, ax = plt.subplots(figsize=a4_dims)
    # sns.heatmap(calendar_return, annot=True, cmap="RdYlGn").set(title="Calendar Heatmap")


    # rolling cagr alpha
    one_year_rolling_cagr_alpha = one_year_rolling_cagr["Value"] - one_year_rolling_cagr["Nifty"]
    three_year_rolling_cagr_alpha = three_year_rolling_cagr["Value"] - three_year_rolling_cagr["Nifty"]
    five_year_rolling_cagr_alpha = five_year_rolling_cagr["Value"] - five_year_rolling_cagr["Nifty"]


    #absolute calendar, quaterly and monthly stats
    #comment below line for run after 1st time
    calendar_return = calendar_return.reset_index()
    absolute_calendar_year_stats, alpha_calendar_return = monthly_quaterly_calendar_return(calendar_return, "Calendar Return")
    # print(absolute_calendar_year_stats)
    # print(alpha_calendar_return)

    absolute_quaterly_return_stats, alpha_quaterly_return = monthly_quaterly_calendar_return(quaterly_return, "Quaterly Return")
    # print(absolute_quaterly_return_stats)
    # print(alpha_quaterly_return)

    absolute_monthly_return_stats, alpha_monthly_return = monthly_quaterly_calendar_return(monthly_return, "Monthly Return")
    # print(absolute_monthly_return_stats)
    # print(alpha_monthly_return)

    alpha_calendar_return_stats = calculate_alpha_stats(alpha_calendar_return, "Alpha Calendar Return Stats")
    # print(alpha_calendar_return_stats)

    alpha_quaterly_return_stats = calculate_alpha_stats(alpha_quaterly_return, "Alpha Quaterly Return Stats")
    # print(alpha_quaterly_return_stats)

    alpha_monthly_return_stats = calculate_alpha_stats(alpha_monthly_return, "Alpha Monthly Return Stats")
    # print(alpha_monthly_return_stats)


    #pre 2020 cagr
    strategy, column_name = calculate_cagr(daily_log_with_nifty[daily_log_with_nifty["Date"] < '2020-01-01'], "Value")
    benchmark, column_name = calculate_cagr(daily_log_with_nifty[daily_log_with_nifty["Date"] < '2020-01-01'], "Nifty")
    pre_return_table = pd.DataFrame(columns=["Strategy", "Benchmark"], index=column_name)
    pre_return_table["Strategy"] = strategy
    pre_return_table["Benchmark"] = benchmark
    # print(pre_return_table)


    #monthly, quaterly and calendar year consecutive win and loss
    consecutive_wins, consecutive_loss = [], []
    consecutive_wins.append(calculate_consecutive_wins_and_loss(monthly_return, True))
    consecutive_wins.append(calculate_consecutive_wins_and_loss(quaterly_return, True))
    consecutive_wins.append(calculate_consecutive_wins_and_loss(calendar_return, True))
    consecutive_loss.append(calculate_consecutive_wins_and_loss(monthly_return, False))
    consecutive_loss.append(calculate_consecutive_wins_and_loss(quaterly_return, False))
    consecutive_loss.append(calculate_consecutive_wins_and_loss(calendar_return, False))
    consecutive_win_loss_dataframe = pd.DataFrame(columns=["Consecutive Wins", "Consecutive Loss"], index=["Monthly", "Quaterly", "Calendar Year"])
    consecutive_win_loss_dataframe["Consecutive Wins"] = consecutive_wins
    consecutive_win_loss_dataframe["Consecutive Loss"] = consecutive_loss
    # print(consecutive_win_loss_dataframe)


    #days to recovery
    # days_to_recover_dataframe = days_to_recover(daily_log_with_nifty.set_index('Date')[{'Value'}], rf=0.07)
    # days_to_recover_dataframe = days_to_recover(daily_log_with_nifty.set_index('Date')[['Value']], rf=0.07)
    # print(days_to_recover_dataframe)
    days_to_recover_dataframe = drawdown_function(daily_log_with_nifty.set_index('Date')[['Value']])


    # #skew kurtosis and geometric mean
    # skew_data = []
    # skew_data.append(one_year_rolling_cagr.skew().to_list())
    # skew_data.append(three_year_rolling_cagr.skew().to_list())
    # skew_data.append(five_year_rolling_cagr.skew().to_list())
    # skew_dataframe = pd.DataFrame(skew_data, columns=["Strategy", "Benchmark"], index=["1 Year Rolling CAGR", "3 Year Rolling CAGR", "5 Year Rolling CAGR"])
    # # print(skew_dataframe)

    # kurtosis_data = []
    # kurtosis_data.append(one_year_rolling_cagr.kurtosis().to_list())
    # kurtosis_data.append(three_year_rolling_cagr.kurtosis().to_list())
    # kurtosis_data.append(five_year_rolling_cagr.kurtosis().to_list())
    # kurtosis_dataframe = pd.DataFrame(kurtosis_data, columns=["Strategy", "Benchmark"], index=["1 Year Rolling CAGR", "3 Year Rolling CAGR", "5 Year Rolling CAGR"])
    # # print(kurtosis_dataframe)

    strategy = []
    strategy.append(calculate_geometric_mean(one_year_rolling_cagr["Value"].dropna()))
    strategy.append(calculate_geometric_mean(three_year_rolling_cagr["Value"].dropna()))
    strategy.append(calculate_geometric_mean(five_year_rolling_cagr["Value"].dropna()))
    benchmark = []
    benchmark.append(calculate_geometric_mean(one_year_rolling_cagr["Nifty"].dropna()))
    benchmark.append(calculate_geometric_mean(three_year_rolling_cagr["Nifty"].dropna()))
    benchmark.append(calculate_geometric_mean(five_year_rolling_cagr["Nifty"].dropna()))
    geometric_mean_dataframe = pd.DataFrame(columns=["Strategy", "Benchmark"], index=["1 Year Rolling CAGR", "3 Year Rolling CAGR", "5 Year Rolling CAGR"])
    geometric_mean_dataframe["Strategy"] = strategy
    geometric_mean_dataframe["Benchmark"] = benchmark
    # print(geometric_mean_dataframe)
    

    #average alpha in positive and negative years
    average_alpha_in_positive_and_negative_years_data = []
    average_alpha_in_positive_and_negative_years_data.append(round(sum(np.where(calendar_return["Benchmark"] > 0, calendar_return["Strategy"] - calendar_return["Benchmark"], 0)) / np.count_nonzero(np.where(calendar_return["Benchmark"] > 0, 1, 0)), 2))
    average_alpha_in_positive_and_negative_years_data.append(round(sum(np.where(calendar_return["Benchmark"] < 0, calendar_return["Strategy"] - calendar_return["Benchmark"], 0)) / np.count_nonzero(np.where(calendar_return["Benchmark"] < 0, 1, 0)), 2))
    average_alpha_in_positive_and_negative_years_dataframe = pd.DataFrame(average_alpha_in_positive_and_negative_years_data,
                                                                        index=["Average Alpha in Positive Years", "Average Alpha in Negative Years"],
                                                                        columns=["Data"])


    #mean of worst five drawdowns and average days to recover data
    # mean_of_worst_five_drawdowns_and_average_days_to_recover_data = []
    # mean_of_worst_five_drawdowns_and_average_days_to_recover_data.append(round(calculate_average_days_to_recover(days_to_recover_dataframe, 4), 2))
    # mean_of_worst_five_drawdowns_and_average_days_to_recover_data.append(round(calculate_average_days_to_recover(days_to_recover_dataframe, 3), 2))
    # mean_of_worst_five_drawdowns_and_average_days_to_recover_dataframe = pd.DataFrame(mean_of_worst_five_drawdowns_and_average_days_to_recover_data,
    #                                                                                 index=["Mean of worst 5 drawdowns", "Average days to recover"],
    #                                                                                 columns=["Data"])



    #continuous underperformance ratio data
    continuous_underperformance_ratio_data = []
    if len(three_year_rolling_cagr.dropna()) == 0:
        continuous_underperformance_ratio_data.append(np.nan)
    else:
        continuous_underperformance_ratio_data.append(round(np.count_nonzero(np.where(three_year_rolling_cagr.dropna()["Value"] < three_year_rolling_cagr.dropna()["Nifty"], 1, 0)) / len(three_year_rolling_cagr.dropna()) * 100, 2))
    if len(five_year_rolling_cagr.dropna()) == 0:
        continuous_underperformance_ratio_data.append(np.nan)
    else:
        continuous_underperformance_ratio_data.append(round(np.count_nonzero(np.where(five_year_rolling_cagr.dropna()["Value"] < five_year_rolling_cagr.dropna()["Nifty"], 1, 0)) / len(five_year_rolling_cagr.dropna()) * 100, 2))
    continuous_underperformance_ratio_dataframe = pd.DataFrame(continuous_underperformance_ratio_data, 
                                                                index=['three year rolling CAGR', 'five year rolling CAGR'], 
                                                                columns=['Continuous Underperformance Ratio'])
    continuous_underperformance_ratio_dataframe



    #negative rolling returns of strategy probability data
    negative_rolling_returns_of_strategy_probability_data = []
    negative_rolling_returns_of_strategy_probability_data.append(sum(three_year_rolling_cagr_probability_range_data[:2]))
    negative_rolling_returns_of_strategy_probability_data.append(sum(five_year_rolling_cagr_probability_range_data[:2]))
    negative_rolling_returns_of_strategy_probability_dataframe = pd.DataFrame(negative_rolling_returns_of_strategy_probability_data,
                                                                            index=['three year rolling CAGR', 'five year rolling CAGR'], 
                                                                            columns=['Negative Rolling Returns of Strategy Probability'])


    #year wise cagr si
    years_list, strategy, benchmark = daily_log_with_nifty['Date'].dt.year.unique(), [], []
    for year in years_list:
        strategy.append(cagr_si(daily_log_with_nifty[daily_log_with_nifty['Date'] <= str(year)+'-12-31'], 'Value'))
        benchmark.append(cagr_si(daily_log_with_nifty[daily_log_with_nifty['Date'] <= str(year)+'-12-31'], 'Nifty'))
    cagr_si_dataframe = pd.DataFrame(columns=[year for year in years_list], index=['CAGR SI strategy', 'CAGR SI benchmark'])
    cagr_si_dataframe.loc['CAGR SI strategy'] = strategy
    cagr_si_dataframe.loc['CAGR SI benchmark'] = benchmark
    cagr_si_dataframe.loc['Alpha'] = cagr_si_dataframe.loc['CAGR SI strategy'] - cagr_si_dataframe.loc['CAGR SI benchmark']
    # print(cagr_si_dataframe)

    #year wise cagr till
    years_list, strategy_till, benchmark_till = daily_log_with_nifty['Date'].dt.year.unique(), [], []
    for year in years_list:
        strategy_till.append(cagr_si(daily_log_with_nifty[daily_log_with_nifty['Date'] >= str(year)+'-01-01'], 'Value'))
        benchmark_till.append(cagr_si(daily_log_with_nifty[daily_log_with_nifty['Date'] >= str(year)+'-01-01'], 'Nifty'))
    cagr_till_dataframe = pd.DataFrame(columns=[year for year in years_list], index=['CAGR Till strategy', 'CAGR Till benchmark'])
    cagr_till_dataframe.loc['CAGR Till strategy'] = strategy_till
    cagr_till_dataframe.loc['CAGR Till benchmark'] = benchmark_till
    cagr_till_dataframe.loc['Alpha'] = cagr_till_dataframe.loc['CAGR Till strategy'] - cagr_till_dataframe.loc['CAGR Till benchmark']
    cagr_till_dataframe = cagr_till_dataframe.round(decimals=2)
    # print(cagr_till_dataframe)

    # CAGR 2010 to 2019
    strategy, column_name = calculate_cagr(daily_log_with_nifty[(daily_log_with_nifty["Date"] >= '2010-01-01') & ((daily_log_with_nifty["Date"] <= '2019-12-31'))], "Value")
    benchmark, column_name = calculate_cagr(daily_log_with_nifty[(daily_log_with_nifty["Date"] >= '2010-01-01') & ((daily_log_with_nifty["Date"] <= '2019-12-31'))], "Nifty")
    cagr_from_2010_to_2019_df = pd.DataFrame(columns=["Strategy", "Benchmark"], index=column_name)
    cagr_from_2010_to_2019_df['Strategy'] = strategy
    cagr_from_2010_to_2019_df['Benchmark'] = benchmark
    # print(cagr_from_2010_to_2019_df)


    # CAGR since 2010
    strategy, column_name = calculate_cagr(daily_log_with_nifty[daily_log_with_nifty["Date"] >= '2010-01-01'], "Value")
    benchmark, column_name = calculate_cagr(daily_log_with_nifty[daily_log_with_nifty["Date"] >= '2010-01-01'], "Nifty")
    cagr_since_2010_df = pd.DataFrame(columns=["Strategy", "Benchmark"], index=column_name)
    cagr_since_2010_df['Strategy'] = strategy
    cagr_since_2010_df['Benchmark'] = benchmark
    # print(cagr_since_2010_df)

    #rolling_sharpe
    rolling_sharpe = []
    rolling_sharpe.append(round(calculate_rolling_sharpe(one_year_rolling_cagr, 1, daily_log_with_nifty).iloc[-1], 2))
    rolling_sharpe.append(round(calculate_rolling_sharpe(three_year_rolling_cagr, 3, daily_log_with_nifty).iloc[-1], 2))
    rolling_sharpe.append(round(calculate_rolling_sharpe(five_year_rolling_cagr, 5, daily_log_with_nifty).iloc[-1], 2))
    rolling_sharpe.append(round(calculate_rolling_sharpe(seven_year_rolling_cagr, 7, daily_log_with_nifty).iloc[-1], 2))
    rolling_sharpe.append(round(calculate_rolling_sharpe(ten_year_rolling_cagr, 10, daily_log_with_nifty).iloc[-1], 2))

    #rolling_std
    rolling_std = []
    rolling_std.append(round(calculate_rolling_std(1, daily_log_with_nifty).iloc[-1], 2))
    rolling_std.append(round(calculate_rolling_std(3, daily_log_with_nifty).iloc[-1], 2))
    rolling_std.append(round(calculate_rolling_std(5, daily_log_with_nifty).iloc[-1], 2))
    rolling_std.append(round(calculate_rolling_std(7, daily_log_with_nifty).iloc[-1], 2))
    rolling_std.append(round(calculate_rolling_std(10, daily_log_with_nifty).iloc[-1], 2))

    #rolling_sharpe
    rolling_sortino = []
    rolling_sortino.append(round(calculate_rolling_sortino(one_year_rolling_cagr, 1, daily_log_with_nifty), 2))
    rolling_sortino.append(round(calculate_rolling_sortino(three_year_rolling_cagr, 3, daily_log_with_nifty), 2))
    rolling_sortino.append(round(calculate_rolling_sortino(five_year_rolling_cagr, 5, daily_log_with_nifty), 2))
    rolling_sortino.append(round(calculate_rolling_sortino(seven_year_rolling_cagr, 7, daily_log_with_nifty), 2))
    rolling_sortino.append(round(calculate_rolling_sortino(ten_year_rolling_cagr, 10, daily_log_with_nifty), 2))

    #rolling metrics
    #rolling 1 year alpha
    one_year_rolling_cagr['Alpha'] = (one_year_rolling_cagr['Value'] - one_year_rolling_cagr['Nifty'])
    rolling_one_year_alpha = (len(one_year_rolling_cagr[one_year_rolling_cagr['Alpha'] > 0]) / len(one_year_rolling_cagr)) * 100
    # rolling 1 year positive
    rolling_one_year_positive = (len(one_year_rolling_cagr[one_year_rolling_cagr['Value'] > 0]) / len(one_year_rolling_cagr)) * 100
    # rolling monthly alpha
    daily_log_with_nifty['value_rolling_monthly'] = daily_log_with_nifty['Value'].pct_change(22)
    daily_log_with_nifty['nifty_rolling_monthly'] = daily_log_with_nifty['Nifty'].pct_change(22)
    daily_log_with_nifty['monthly_alpha'] = daily_log_with_nifty['value_rolling_monthly'] - daily_log_with_nifty['nifty_rolling_monthly']
    rolling_monthly_alpha = (len(daily_log_with_nifty[daily_log_with_nifty['monthly_alpha'] > 0]) / len(daily_log_with_nifty['monthly_alpha'].dropna())) * 100
    # rolling 2 month alpha
    daily_log_with_nifty['value_rolling_2month'] = daily_log_with_nifty['Value'].pct_change(22*2)
    daily_log_with_nifty['nifty_rolling_2month'] = daily_log_with_nifty['Nifty'].pct_change(22*2)
    daily_log_with_nifty['2month_alpha'] = daily_log_with_nifty['value_rolling_2month'] - daily_log_with_nifty['nifty_rolling_2month']
    rolling_2month_alpha = (len(daily_log_with_nifty[daily_log_with_nifty['2month_alpha'] > 0]) / len(daily_log_with_nifty['2month_alpha'].dropna())) * 100
    rolling_2month_alpha
    # rolling quaterly alpha
    quaterly_rolling_cagr['Alpha'] = quaterly_rolling_cagr['Value'] - quaterly_rolling_cagr['Nifty']
    rolling_quaterly_alpha = (len(quaterly_rolling_cagr[quaterly_rolling_cagr['Alpha'] > 0]) / len(quaterly_rolling_cagr)) * 100

    #longest flat time (number of days to recover from previous max)
    daily_log_with_nifty['cummax'] = daily_log_with_nifty['Value'].cummax()
    temp_df = daily_log_with_nifty.drop_duplicates('cummax')
    temp_df['Date_diff'] = temp_df['Date'].diff()
    temp_df['Date_shift'] = temp_df['Date'].shift(1)
    temp_df['Value_shift'] = temp_df['Value'].shift(1)
    final_df = temp_df[temp_df['Date_diff'] == temp_df['Date_diff'].max()]
    final_df = final_df[['Date', 'Value', 'Date_diff', 'Date_shift', 'Value_shift']].reset_index(drop=True)
    final_df.rename(columns={'Date': 'New Date', 'Value': 'New Value', 'Date_diff': 'Number Of Days', 'Date_shift': 'Previous Date',
                            'Value_shift':'Previous Value'}, inplace=True)
    final_df = final_df.round(2)
    longest_flat_time = final_df['Number Of Days'][0].days

    #max drawdown recovery days(recovery days for max drawdown)
    # print(days_to_recover_dataframe)
    max_drawdown_recovery_days = days_to_recover_dataframe.loc[1, 'Recovery Days'].days

    #mean_of_worst_five_drawdowns
    mean_of_worst_five_drawdowns = round(days_to_recover_dataframe['Max Drawdown'].mean(), 2)

    #average_days_to_recover
    average_days_to_recover = days_to_recover_dataframe['Recovery Days'].mean().days
    # print(days_to_recover_dataframe)
    # return

    data = {}
    # data["Strategy"] = np.nan
    #1. No of years + ve / total no of years
    data["No of years + ve / total no of years"] = absolute_calendar_year_stats["Strategy Calendar Return"]["Positive Calendar Return Percentage"]
    #2. No of years + ve Alpha / Total no of years
    data["No of years + ve Alpha / Total no of years"] = alpha_calendar_return_stats["Alpha Calendar Return Stats"]["Positive Alpha Calendar Return Stats Percentage"]
    #3. Avg + ve
    data["Avg + ve"] = absolute_calendar_year_stats["Strategy Calendar Return"]["Mean Of Positives"]
    # 4. Avg + alpha
    data["Avg + alpha"] = alpha_calendar_return_stats["Alpha Calendar Return Stats"]["Mean Of Positives"]
    #5. Avg - ve
    data["Avg - ve"] = absolute_calendar_year_stats["Strategy Calendar Return"]["Mean Of Negatives"]
    # 6. Avg - alpha
    data["Avg - alpha"] = alpha_calendar_return_stats["Alpha Calendar Return Stats"]["Mean Of Negatives"]
    # 7. Probability of 10 percent rolling 3 years
    data["Probability of 10 percent rolling 3 years"] = sum(three_year_rolling_cagr_probability_chart.loc["3 Year Rolling CAGR"][3:])
    # 8. Probability of 20 percent rolling 3 years
    data["Probability of 20 percent rolling 3 years"] = sum(three_year_rolling_cagr_probability_chart.loc["3 Year Rolling CAGR"][4:])
    # 9. Probablity of 30 percent rolling 3 years
    data["Probability of 30 percent rolling 3 years"] = sum(three_year_rolling_cagr_probability_chart.loc["3 Year Rolling CAGR"][5:])
    # 10. Max drawdown
    data["Max drawdown"] = risk_ratios["Strategy"]["Max Drawdown"]
    # 11. Stdev of returns
    data["Stdev of returns"] = risk_ratios["Strategy"]["STD"]
    # 12.0 Beta
    data["Beta"] = risk_ratios["Strategy"]["Beta"]
    # 12. Sharpe
    data["Sharpe"] = risk_ratios["Strategy"]["Sharpe"]
    # 13. Sortino
    data["Sortino"] = risk_ratios["Strategy"]["Sortino"]
    # 13.1 Information Ratio
    data["Information Ratio"] = risk_ratios["Strategy"]["Information Ratio"]
    # 13.2 upside capture ratio
    data["Upside Capture Ratio"] = risk_ratios["Strategy"]["Upside Capture Ratio"]
    #13.3 downside capture ratio
    data["Downside Capture Ratio"] = risk_ratios["Strategy"]["Downside Capture Ratio"]
    #13.4 r_square
    data["R Square"] = r_square
    # 14. CAGR SI
    data["CAGR SI"] = return_table["Strategy"]["CAGR SI"]
    # 15.0 CAGR 15
    data["CAGR 15"] = return_table["Strategy"]["CAGR 15"]
    # 15.1 CAGR 10
    data["CAGR 10"] = return_table["Strategy"]["CAGR 10"]
    # 16. CAGR 7
    data["CAGR 7"] = return_table["Strategy"]["CAGR 7"]
    # 17. CAGR 5
    data["CAGR 5"] = return_table["Strategy"]["CAGR 5"]
    # 18. CAGR 3
    data["CAGR 3"] = return_table["Strategy"]["CAGR 3"]
    # 19.0 CAGR 2
    data["CAGR 2"] = return_table["Strategy"]["CAGR 2"]
    # 19.1 CAGR 1
    data["CAGR 1"] = return_table["Strategy"]["CAGR 1"]
    # 20 CAGR 6m
    data["CAGR 6m"] = return_table["Strategy"]["6 Month"]
    # 20.2 CAGR 3m
    data["CAGR 3m"] = return_table["Strategy"]["3 Month"]
    # 20.3 CAGR 2m
    data["CAGR 2m"] = return_table["Strategy"]["2 Month"]
    # 20.4 CAGR 1m
    data["CAGR 1m"] = return_table["Strategy"]["1 Month"]
    # 20.5 xtimes 1
    data["xtimes 1"] = return_table["Strategy"]["xtimes 1"]
    # 20.6 xtimes 2
    data["xtimes 2"] = return_table["Strategy"]["xtimes 2"]
    # 20.5 xtimes 3
    data["xtimes 3"] = return_table["Strategy"]["xtimes 3"]
    # 20.5 xtimes 5
    data["xtimes 5"] = return_table["Strategy"]["xtimes 5"]
    # 20.5 xtimes 7
    data["xtimes 7"] = return_table["Strategy"]["xtimes 7"]
    # 20.5 xtimes 10
    data["xtimes 10"] = return_table["Strategy"]["xtimes 10"]
    # 20.5 xtimes 15
    data["xtimes 15"] = return_table["Strategy"]["xtimes 15"]
    # 20.5 xtimes 16
    data["xtimes SI"] = return_table["Strategy"]["xtimes SI"]
    # 21. Avg Alpha in positive years
    data["Avg Alpha in positive years"] = average_alpha_in_positive_and_negative_years_dataframe["Data"]["Average Alpha in Positive Years"]
    # 22. Avg Alpha in negative years
    data["Avg Alpha in negative years"] = average_alpha_in_positive_and_negative_years_dataframe["Data"]["Average Alpha in Negative Years"]
    # 23. Mean of worst 5 drawdowns
    data["Mean of worst 5 drawdowns"] = mean_of_worst_five_drawdowns
    # 24. Avg days to recovery
    data["Avg days to recovery"] = average_days_to_recover
    # 33. Alpha ratio(calendar alpha percentage)
    data["Calendar Alpha percentage"] = alpha_percentage["Alpha Ratio"]["Calendar"]
    # 33.1 Alpha ratio(quaterly alpha percentage)
    data["Quaterly Alpha percentage"] = alpha_percentage['Alpha Ratio']['Quaterly']
    # 33.2 Alpha ratio(monthly alpha percentage)
    data["Monthly Alpha percentage"] = alpha_percentage['Alpha Ratio']['Monthly']
    # 33.4. Absolute Positive(calendar alpha percentage)
    data["Calendar Absolute Positive"] = alpha_percentage["Absolute Positive"]["Calendar"]
    # 33.5 Absolute Positive(quaterly alpha percentage)
    data["Quaterly Absolute Positive"] = alpha_percentage['Absolute Positive']['Quaterly']
    # 33.6 Absolute Positive(monthly alpha percentage)
    data["Monthly Absolute Positive"] = alpha_percentage['Absolute Positive']['Monthly']
    # 38. mean quaterly +ve alpha
    data["Mean Quaterly +ve Alpha"] = alpha_quaterly_return_stats["Alpha Quaterly Return Stats"]["Mean Of Positives"]
    # 39. mean quaterly -ve alpha
    data["Mean Quaterly -ve Alpha"] = alpha_quaterly_return_stats["Alpha Quaterly Return Stats"]["Mean Of Negatives"]
    # 40. Treynor ratio
    data["Treynor ratio"] = risk_ratios["Strategy"]["Treynor Ratio"]
    # 41. Jensen Alpha Ratio
    data["Jensen Alpha Ratio"] = risk_ratios["Strategy"]["Jensen Alpha"]
    # 42 Continuous underperformance ratio - period of time strategy underperforms benchmark on a daily basis - three year rolling cagr
    data["Continuous underperformance ratio three year rolling cagr"] = continuous_underperformance_ratio_dataframe["Continuous Underperformance Ratio"]["three year rolling CAGR"]
    # 42.1 Continuous underperformance ratio - period of time strategy underperforms benchmark on a daily basis - five year rolling cagr
    data["Continuous underperformance ratio five year rolling cagr"] = continuous_underperformance_ratio_dataframe["Continuous Underperformance Ratio"]["five year rolling CAGR"]
    # 45. Negative Rolling returns of strategy Probability - three year rolling cagr
    data["Negative Rolling returns three year rolling cagr"] = negative_rolling_returns_of_strategy_probability_dataframe["Negative Rolling Returns of Strategy Probability"]["three year rolling CAGR"]
    # 45.1 Negative Rolling returns of strategy Probability - five year rolling cagr
    data["Negative Rolling returns five year rolling cagr"] = negative_rolling_returns_of_strategy_probability_dataframe["Negative Rolling Returns of Strategy Probability"]["five year rolling CAGR"]
    # 1 year rolling cagr mean
    data["One Year Rolling CAGR Mean"] = rolling_returns['Mean']['1 Year Rolling CAGR']
    # 3 year rolling cagr mean
    data["Three Year Rolling CAGR Mean"] = rolling_returns['Mean']['3 Year Rolling CAGR']
    # 5 year rolling cagr mean
    data["Five Year Rolling CAGR Mean"] = rolling_returns['Mean']['5 Year Rolling CAGR']
    #rolling sharpe 1, 3, 5, 7, 10
    sharpe_index = [1, 3, 5, 7, 10]
    for index in range(len(rolling_sharpe)):
        data[f'Sharpe {sharpe_index[index]}'] = rolling_sharpe[index]
    #rolling std 1, 3, 5, 7, 10
    rolling_std_index = [1, 3, 5, 7, 10]
    for index in range(len(rolling_std)):
        data[f'STD {rolling_std_index[index]}'] = rolling_std[index]
    #rolling_sortino 1, 3, 5, 7, 10
    sortino_index = [1, 3, 5, 7, 10]
    for index in range(len(rolling_sortino)):
        data[f'Sortino {sortino_index[index]}'] = rolling_sortino[index]
    #cagr si from first year
    cagr_si_dataframe_values = cagr_si_dataframe.loc['CAGR SI strategy'].to_list()
    cagr_si_dataframe_index = ['CAGR Till '+str(year) for year in list(cagr_si_dataframe.columns)]
    for index in range(len(cagr_si_dataframe_values)):
        data[cagr_si_dataframe_index[index]] = cagr_si_dataframe_values[index]
    #cagr si till last year
    cagr_till_dataframe_values = cagr_till_dataframe.loc['CAGR Till strategy'].to_list()
    cagr_till_dataframe_index = ['CAGR Since '+str(year) for year in list(cagr_till_dataframe.columns)]
    for index in range(len(cagr_till_dataframe_values)):
        data[cagr_till_dataframe_index[index]] = cagr_till_dataframe_values[index]
    # calendar year return from 2006
    calendar_return_values = calendar_return['Strategy'].to_list()
    calendar_return_index = ['CY '+ str(year) for year in calendar_return['Date'].to_list()]
    for index in range(len(calendar_return_values)):
        data[calendar_return_index[index]] = calendar_return_values[index]
    #CYD
    data['CYTD'] = CYTD
    #FTD
    data['FYTD'] = FYTD
    #closing nav
    data['Closing NAV'] = closing_nav
    # rolling 1 year alpha
    data['Rolling 1 Year Alpha'] = rolling_one_year_alpha
    # rolling 1 year positive
    data['Rolling 1 Year Positive'] = rolling_one_year_positive
    # rolling monthly alpha
    data['Rolling Monthly Alpha'] = rolling_monthly_alpha
    # rolling 2 month alpha
    data['Rolling 2 Month Alpha'] = rolling_2month_alpha
    # rolling quaterly alpha
    data['Rolling Quarterly Alpha'] = rolling_quaterly_alpha
    #longest flat time
    data['Longest Flat Time in Days'] = longest_flat_time
    #max drawdown recovery days
    data['Max Drawdown Recovery Days'] = max_drawdown_recovery_days

    # return data, cagr_si_dataframe, calendar_return
    return data, calendar_return
