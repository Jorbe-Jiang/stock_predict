from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd
import numpy as np

from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import AbstractHolidayCalendar, nearest_workday, Holiday


class myCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('RPC National Day', month=10, day=1),
        Holiday('RPC National Day', month=10, day=2),
        Holiday('RPC National Day', month=10, day=3),
        Holiday('RPC National Day', month=10, day=4),
        Holiday('RPC National Day', month=10, day=5),
        Holiday('RPC National Day', month=10, day=6),
        Holiday('RPC National Day', month=10, day=7),
        Holiday('Labour Day', month=5, day=1),
        Holiday('Labour Day', month=5, day=2),
        Holiday('Labour Day', month=5, day=3),
        Holiday('New year', month=1, day=1),
        Holiday('Christmas Day', month=12, day=24),
        Holiday('Christmas Day', month=12, day=25)
    ]


# 记录每个节日的第一天日期，用于后续创建日期特征
festival_start_dates = ['10/1', '5/1', '1/1', '12/24']

myc = CustomBusinessDay(calendar=myCalendar())


def scale_value(value, dataset_min_value, dataset_max_value, min=0.0, max=1.0):
    # 把最小和最大值分别进行缩小和放大，因为考虑到测试集的最小和最大值不一定在训练集的数据范围之内
    dataset_min_value = dataset_min_value * 0.75
    dataset_max_value = dataset_max_value * 1.25
    scale = (max - min) / (dataset_max_value - dataset_min_value)
    scaled_value = scale * value + min - dataset_min_value * scale
    return scaled_value


def inverse_scale_value(scaled_value, dataset_min_value, dataset_max_value, min=0.0, max=1.0):
    # 把最小和最大值分别进行缩小和放大，因为考虑到测试集的最小和最大值不一定在训练集的数据范围之内
    dataset_min_value = dataset_min_value * 0.75
    dataset_max_value = dataset_max_value * 1.25
    scale = (max - min) / (dataset_max_value - dataset_min_value)
    value = (scaled_value + dataset_min_value * scale - min) / scale
    return value


def create_is_holiday_periods_feature(year, month, day, periods=5):
    """
    三种类别: 给定日期在节日日期前periods天范围内；给定日期在节日日期后periods天范围内；others
    分别对应flag：1，2，0
    """
    str_date = str(year) + '-' + str(month) + '-' + str(day)
    curr_date = pd.Timestamp(str_date)
    for fs_start_date in festival_start_dates:
        fs_start_date = fs_start_date + '/' + str(year)
        roll_forward_days = pd.bdate_range(start=fs_start_date, periods=periods, freq=myc)
        roll_backward_days = pd.bdate_range(end=fs_start_date, periods=periods, freq=myc)
        if curr_date in roll_backward_days:
            return 1
        elif curr_date in roll_forward_days:
            return 2
        else:
            continue
    return 0


def load_raw_data(model_configs, data_dir):
    data_file = os.path.join(data_dir, 'stock_data.csv')
    df = pd.read_csv(data_file)

    wp = open(os.path.join(data_dir, 'prices_min_max_value.txt'), 'w')

    print(df.head())

    df['StockDate'] = pd.to_datetime(df.StockDate, format='%Y-%m-%d')
    df.index = df['StockDate']

    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['StockDate', 'open', 'high', 'low', 'vol', 'amount', 'close'])

    for i in range(0, len(data)):
        new_data['StockDate'][i] = data['StockDate'][i]
        new_data['open'][i] = data['open'][i]
        new_data['high'][i] = data['high'][i]
        new_data['low'][i] = data['low'][i]
        new_data['vol'][i] = data['vol'][i]
        new_data['amount'][i] = data['amount'][i]
        new_data['close'][i] = data['close'][i]

    new_data.index = new_data.StockDate

    dataset = new_data.values

    dataset_open_price_max_value = dataset.max(axis=0)[1]
    dataset_high_price_max_value = dataset.max(axis=0)[2]
    dataset_low_price_max_value = dataset.max(axis=0)[3]
    dataset_vol_price_max_value = dataset.max(axis=0)[4]
    dataset_amount_price_max_value = dataset.max(axis=0)[5]
    dataset_close_price_max_value = dataset.max(axis=0)[6]

    dataset_open_price_min_value = dataset.min(axis=0)[1]
    dataset_high_price_min_value = dataset.min(axis=0)[2]
    dataset_low_price_min_value = dataset.min(axis=0)[3]
    dataset_vol_price_min_value = dataset.min(axis=0)[4]
    dataset_amount_price_min_value = dataset.min(axis=0)[5]
    dataset_close_price_min_value = dataset.min(axis=0)[6]

    wp.write('dataset_open_price_max_value: {}, dataset_open_price_min_value: {}\n'.format(dataset_open_price_max_value,
                                                                                           dataset_open_price_min_value))
    wp.write('dataset_high_price_max_value: {}, dataset_high_price_min_value: {}\n'.format(dataset_high_price_max_value,
                                                                                           dataset_high_price_min_value))
    wp.write('dataset_low_price_max_value: {}, dataset_low_price_min_value: {}\n'.format(dataset_low_price_max_value,
                                                                                         dataset_low_price_min_value))
    wp.write('dataset_vol_price_max_value: {}, dataset_vol_price_min_value: {}\n'.format(dataset_vol_price_max_value,
                                                                                         dataset_vol_price_min_value))
    wp.write('dataset_amount_price_max_value: {}, dataset_amount_price_min_value: {}\n'.format(dataset_amount_price_max_value,
                                                                                               dataset_amount_price_min_value))
    wp.write('dataset_close_price_max_value: {}, dataset_close_price_min_value: {}\n'.format(dataset_close_price_max_value,
                                                                                             dataset_close_price_min_value))
    wp.close()

    # [N]
    scaled_open_price_data = scale_value(dataset[:, 1], dataset_open_price_min_value, dataset_open_price_max_value,
                                         min=0.0, max=1.0)
    scaled_high_price_data = scale_value(dataset[:, 2], dataset_high_price_min_value, dataset_high_price_max_value,
                                         min=0.0, max=1.0)
    scaled_low_price_data = scale_value(dataset[:, 3], dataset_low_price_min_value, dataset_low_price_max_value,
                                        min=0.0, max=1.0)
    scaled_vol_data = scale_value(dataset[:, 4], dataset_vol_price_min_value, dataset_vol_price_max_value,
                                  min=0.0, max=1.0)
    scaled_amount_data = scale_value(dataset[:, 5], dataset_amount_price_min_value, dataset_amount_price_max_value,
                                     min=0.0, max=1.0)
    scaled_close_price_data = scale_value(dataset[:, 6], dataset_close_price_min_value, dataset_close_price_max_value,
                                          min=0.0, max=1.0)

    reshape_scaled_open_price_data = np.reshape(scaled_open_price_data, [-1, 1])
    reshape_scaled_high_price_data = np.reshape(scaled_high_price_data, [-1, 1])
    reshape_scaled_vol_price_data = np.reshape(scaled_vol_data, [-1, 1])
    reshape_scaled_low_price_data = np.reshape(scaled_low_price_data, [-1, 1])
    reshape_scaled_amount_price_data = np.reshape(scaled_amount_data, [-1, 1])
    reshape_scaled_close_price_data = np.reshape(scaled_close_price_data, [-1, 1])

    # 合并价格信息, [N, num_price_types_used], 并确保需要预测的价格在第一维度
    if model_configs.pred_price_type == 'close':
        scaled_target_price_data = scaled_close_price_data
        target_dataset_price_min_value = dataset_close_price_min_value
        target_dataset_price_max_value = dataset_close_price_max_value
        if model_configs.use_vol_amount:
            scaled_concat_price_data = np.concatenate(
                (reshape_scaled_close_price_data, reshape_scaled_open_price_data, reshape_scaled_high_price_data,
                 reshape_scaled_low_price_data,
                 reshape_scaled_vol_price_data, reshape_scaled_amount_price_data), axis=1)
        else:
            scaled_concat_price_data = np.concatenate(
                (reshape_scaled_close_price_data, reshape_scaled_open_price_data, reshape_scaled_high_price_data,
                 reshape_scaled_low_price_data), axis=1)
    elif model_configs.pred_price_type == 'high':
        scaled_target_price_data = scaled_high_price_data
        target_dataset_price_min_value = dataset_high_price_min_value
        target_dataset_price_max_value = dataset_high_price_max_value
        if model_configs.use_vol_amount:
            scaled_concat_price_data = np.concatenate(
                (reshape_scaled_high_price_data, reshape_scaled_open_price_data, reshape_scaled_close_price_data,
                 reshape_scaled_low_price_data,
                 reshape_scaled_vol_price_data, reshape_scaled_amount_price_data), axis=1)
        else:
            scaled_concat_price_data = np.concatenate(
                (reshape_scaled_high_price_data, reshape_scaled_open_price_data, reshape_scaled_close_price_data,
                 reshape_scaled_low_price_data), axis=1)
    elif model_configs.pred_price_type == 'low':
        scaled_target_price_data = scaled_low_price_data
        target_dataset_price_min_value = dataset_low_price_min_value
        target_dataset_price_max_value = dataset_low_price_max_value
        if model_configs.use_vol_amount:
            scaled_concat_price_data = np.concatenate(
                (reshape_scaled_low_price_data, reshape_scaled_open_price_data, reshape_scaled_high_price_data,
                 reshape_scaled_close_price_data,
                 reshape_scaled_vol_price_data, reshape_scaled_amount_price_data), axis=1)
        else:
            scaled_concat_price_data = np.concatenate(
                (reshape_scaled_low_price_data, reshape_scaled_open_price_data, reshape_scaled_high_price_data,
                 reshape_scaled_close_price_data), axis=1)
    elif model_configs.pred_price_type == 'open':
        scaled_target_price_data = scaled_open_price_data
        target_dataset_price_min_value = dataset_open_price_min_value
        target_dataset_price_max_value = dataset_open_price_max_value
        if model_configs.use_vol_amount:
            scaled_concat_price_data = np.concatenate(
                (reshape_scaled_open_price_data, reshape_scaled_close_price_data, reshape_scaled_high_price_data,
                 reshape_scaled_low_price_data,
                 reshape_scaled_vol_price_data, reshape_scaled_amount_price_data), axis=1)
        else:
            scaled_concat_price_data = np.concatenate(
                (reshape_scaled_open_price_data, reshape_scaled_close_price_data, reshape_scaled_high_price_data,
                 reshape_scaled_low_price_data), axis=1)
    else:
        raise NotImplementedError

    assert len(scaled_target_price_data) == len(dataset)
    assert len(scaled_concat_price_data) == len(scaled_target_price_data)
    all_years = [dataset[:, 0][i].year for i in range(len(dataset))]
    all_months = [dataset[:, 0][i].month for i in range(len(dataset))]
    all_days = [dataset[:, 0][i].day for i in range(len(dataset))]
    # 输出这一天是周中的第几天，Monday=0, Sunday=6
    all_weekdays = [dataset[:, 0][i].dayofweek for i in range(len(dataset))]
    # 判断是否在节日日期前后，生成特征: 1, 2, 0
    all_festival_periods_types = [create_is_holiday_periods_feature(all_years[i], all_months[i], all_days[i],
                                                                    periods=model_configs.festival_periods) for i in
                                  range(len(dataset))]

    # 最新的数据作为验证集
    valid_years = all_years[len(dataset) - model_configs.num_valid:]
    train_years = all_years[0: len(dataset) - model_configs.num_valid][-model_configs.num_train:]
    valid_months = all_months[len(dataset) - model_configs.num_valid:]
    train_months = all_months[0: len(dataset) - model_configs.num_valid][-model_configs.num_train:]
    valid_days = all_days[len(dataset) - model_configs.num_valid:]
    train_days = all_days[0: len(dataset) - model_configs.num_valid][-model_configs.num_train:]
    valid_weekdays = all_weekdays[len(dataset) - model_configs.num_valid:]
    train_weekdays = all_weekdays[0: len(dataset) - model_configs.num_valid][-model_configs.num_train:]
    valid_festival_periods_types = all_festival_periods_types[len(dataset) - model_configs.num_valid:]
    train_festival_periods_types = all_festival_periods_types[0: len(dataset) - model_configs.num_valid][-model_configs.num_train:]

    train_history_prices = scaled_concat_price_data[0: len(dataset) - model_configs.num_valid][-model_configs.num_train:]
    valid_history_prices = scaled_concat_price_data[len(dataset) - model_configs.num_valid:]

    train_target_prices = scaled_target_price_data[0: len(dataset) - model_configs.num_valid][-model_configs.num_train:]
    valid_target_prices = scaled_target_price_data[len(dataset) - model_configs.num_valid:]

    train_year_inputs, train_month_inputs, train_day_inputs, train_weekday_inputs, train_festival_periods_types_inputs = [], [], [], [], []
    train_dec_year_inputs, train_dec_month_inputs, train_dec_day_inputs, train_dec_weekday_inputs, train_dec_festival_periods_types_inputs = [], [], [], [], []
    train_targets = []
    train_targets_concat_prices = []  # 用于后续解码器price embedding map function 损失函数计算
    history_prices_for_train = []
    for i in range(model_configs.time_seq_length, len(train_days) - model_configs.max_pred_days):
        history_prices_for_train.append(train_history_prices[i - model_configs.time_seq_length:i])
        train_year_inputs.append(train_years[i - model_configs.time_seq_length:i])
        train_month_inputs.append(train_months[i - model_configs.time_seq_length:i])
        train_day_inputs.append(train_days[i - model_configs.time_seq_length:i])
        train_weekday_inputs.append(train_weekdays[i - model_configs.time_seq_length:i])
        train_festival_periods_types_inputs.append(train_festival_periods_types[i - model_configs.time_seq_length:i])
        train_dec_year_inputs.append(train_years[i:i + model_configs.max_pred_days])
        train_dec_month_inputs.append(train_months[i:i + model_configs.max_pred_days])
        train_dec_day_inputs.append(train_days[i:i + model_configs.max_pred_days])
        train_dec_weekday_inputs.append(train_weekdays[i:i + model_configs.max_pred_days])
        train_dec_festival_periods_types_inputs.append(train_festival_periods_types[i:i + model_configs.max_pred_days])
        train_targets.append(train_target_prices[i:i + model_configs.max_pred_days])
        train_targets_concat_prices.append(train_history_prices[i:i + model_configs.max_pred_days])

    valid_year_inputs, valid_month_inputs, valid_day_inputs, valid_weekday_inputs, valid_festival_periods_types_inputs = [], [], [], [], []
    valid_dec_year_inputs, valid_dec_month_inputs, valid_dec_day_inputs, valid_dec_weekday_inputs, valid_dec_festival_periods_types_inputs = [], [], [], [], []
    valid_targets = []
    valid_targets_concat_prices = []  # 用于后续解码器price embedding map function 损失函数计算
    history_prices_for_valid = []
    for i in range(model_configs.time_seq_length, len(valid_days) - model_configs.max_pred_days):
        history_prices_for_valid.append(valid_history_prices[i - model_configs.time_seq_length:i])
        valid_year_inputs.append(valid_years[i - model_configs.time_seq_length:i])
        valid_month_inputs.append(valid_months[i - model_configs.time_seq_length:i])
        valid_day_inputs.append(valid_days[i - model_configs.time_seq_length:i])
        valid_weekday_inputs.append(valid_weekdays[i - model_configs.time_seq_length:i])
        valid_festival_periods_types_inputs.append(valid_festival_periods_types[i - model_configs.time_seq_length:i])
        valid_dec_year_inputs.append(valid_years[i:i + model_configs.max_pred_days])
        valid_dec_month_inputs.append(valid_months[i:i + model_configs.max_pred_days])
        valid_dec_day_inputs.append(valid_days[i:i + model_configs.max_pred_days])
        valid_dec_weekday_inputs.append(valid_weekdays[i:i + model_configs.max_pred_days])
        valid_dec_festival_periods_types_inputs.append(valid_festival_periods_types[i:i + model_configs.max_pred_days])
        valid_targets.append(valid_target_prices[i:i + model_configs.max_pred_days])
        valid_targets_concat_prices.append(valid_history_prices[i:i + model_configs.max_pred_days])

    return (history_prices_for_train, train_month_inputs, train_day_inputs, train_weekday_inputs, train_festival_periods_types_inputs, train_dec_month_inputs, train_dec_day_inputs, train_dec_weekday_inputs, train_dec_festival_periods_types_inputs, train_targets, train_year_inputs, train_dec_year_inputs, train_targets_concat_prices), \
           (history_prices_for_valid, valid_month_inputs, valid_day_inputs, valid_weekday_inputs, valid_festival_periods_types_inputs, valid_dec_month_inputs, valid_dec_day_inputs, valid_dec_weekday_inputs, valid_dec_festival_periods_types_inputs, valid_targets, valid_year_inputs, valid_dec_year_inputs, valid_targets_concat_prices), \
           target_dataset_price_min_value, target_dataset_price_max_value


def create_future_dates(last_year, last_month, last_day, future_num_days=1):
    """
    根据当前日期，生成未来future_num_days的工作日日期（可能包含节日）
    参考：https://segmentfault.com/a/1190000011145901   https://www.cnblogs.com/rachelross/p/10487021.html
    """
    str_date = str(last_month)+'/'+str(last_day)+'/'+str(last_year)

    future_dates = pd.bdate_range(start=str_date, periods=future_num_days*2, freq=myc)
    future_years = [date.year for date in future_dates][1:future_num_days+1]  # 不包含当天
    future_months = [date.month for date in future_dates][1:future_num_days+1]
    future_days = [date.day for date in future_dates][1:future_num_days+1]
    future_weekdays = [date.dayofweek for date in future_dates][1:future_num_days+1]
    return future_years, future_months, future_days, future_weekdays


def load_test_raw_data(model_configs, data_dir):
    data_file = os.path.join(data_dir, 'stock_data.csv')
    df = pd.read_csv(data_file)

    print(df.head())

    df['StockDate'] = pd.to_datetime(df.StockDate, format='%Y-%m-%d')
    df.index = df['StockDate']

    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['StockDate', 'open', 'high', 'low', 'vol', 'amount', 'close'])

    for i in range(0, len(data)):
        new_data['StockDate'][i] = data['StockDate'][i]
        new_data['open'][i] = data['open'][i]
        new_data['high'][i] = data['high'][i]
        new_data['low'][i] = data['low'][i]
        new_data['vol'][i] = data['vol'][i]
        new_data['amount'][i] = data['amount'][i]
        new_data['close'][i] = data['close'][i]

    new_data.index = new_data.StockDate

    dataset = new_data.values

    with open(os.path.join(data_dir, 'prices_min_max_value.txt'), 'r') as fp:
        for idx, line in enumerate(fp.readlines()):
            line = line.strip()
            s_max, s_min = line.split(',')
            if 'open' in s_max:
                dataset_open_price_max_value = float(s_max[s_max.index(':')+1:].strip())
                dataset_open_price_min_value = float(s_min[s_min.index(':')+1:].strip())
            elif 'high' in s_max:
                dataset_high_price_max_value = float(s_max[s_max.index(':') + 1:].strip())
                dataset_high_price_min_value = float(s_min[s_min.index(':') + 1:].strip())
            elif 'low' in s_max:
                dataset_low_price_max_value = float(s_max[s_max.index(':') + 1:].strip())
                dataset_low_price_min_value = float(s_min[s_min.index(':') + 1:].strip())
            elif 'vol' in s_max:
                dataset_vol_price_max_value = float(s_max[s_max.index(':') + 1:].strip())
                dataset_vol_price_min_value = float(s_min[s_min.index(':') + 1:].strip())
            elif 'amount' in s_max:
                dataset_amount_price_max_value = float(s_max[s_max.index(':') + 1:].strip())
                dataset_amount_price_min_value = float(s_min[s_min.index(':') + 1:].strip())
            elif 'close' in s_max:
                dataset_close_price_max_value = float(s_max[s_max.index(':') + 1:].strip())
                dataset_close_price_min_value = float(s_min[s_min.index(':') + 1:].strip())

    # [N]
    scaled_open_price_data = scale_value(dataset[:, 1], dataset_open_price_min_value, dataset_open_price_max_value,
                                         min=0.0, max=1.0)
    scaled_high_price_data = scale_value(dataset[:, 2], dataset_high_price_min_value, dataset_high_price_max_value,
                                         min=0.0, max=1.0)
    scaled_low_price_data = scale_value(dataset[:, 3], dataset_low_price_min_value, dataset_low_price_max_value,
                                        min=0.0, max=1.0)
    scaled_vol_data = scale_value(dataset[:, 4], dataset_vol_price_min_value, dataset_vol_price_max_value,
                                  min=0.0, max=1.0)
    scaled_amount_data = scale_value(dataset[:, 5], dataset_amount_price_min_value, dataset_amount_price_max_value,
                                     min=0.0, max=1.0)
    scaled_close_price_data = scale_value(dataset[:, 6], dataset_close_price_min_value, dataset_close_price_max_value,
                                          min=0.0, max=1.0)

    reshape_scaled_open_price_data = np.reshape(scaled_open_price_data, [-1, 1])
    reshape_scaled_high_price_data = np.reshape(scaled_high_price_data, [-1, 1])
    reshape_scaled_vol_price_data = np.reshape(scaled_vol_data, [-1, 1])
    reshape_scaled_low_price_data = np.reshape(scaled_low_price_data, [-1, 1])
    reshape_scaled_amount_price_data = np.reshape(scaled_amount_data, [-1, 1])
    reshape_scaled_close_price_data = np.reshape(scaled_close_price_data, [-1, 1])

    # 合并价格信息, [N, num_price_types_used], 并确保需要预测的价格在第一维度
    if model_configs.pred_price_type == 'close':
        scaled_target_price_data = scaled_close_price_data
        target_dataset_price_min_value = dataset_close_price_min_value
        target_dataset_price_max_value = dataset_close_price_max_value
        if model_configs.use_vol_amount:
            scaled_concat_price_data = np.concatenate(
                (reshape_scaled_close_price_data, reshape_scaled_open_price_data, reshape_scaled_high_price_data,
                 reshape_scaled_low_price_data,
                 reshape_scaled_vol_price_data, reshape_scaled_amount_price_data), axis=1)
        else:
            scaled_concat_price_data = np.concatenate(
                (reshape_scaled_close_price_data, reshape_scaled_open_price_data, reshape_scaled_high_price_data,
                 reshape_scaled_low_price_data), axis=1)
    elif model_configs.pred_price_type == 'high':
        scaled_target_price_data = scaled_high_price_data
        target_dataset_price_min_value = dataset_high_price_min_value
        target_dataset_price_max_value = dataset_high_price_max_value
        if model_configs.use_vol_amount:
            scaled_concat_price_data = np.concatenate(
                (reshape_scaled_high_price_data, reshape_scaled_open_price_data, reshape_scaled_close_price_data,
                 reshape_scaled_low_price_data,
                 reshape_scaled_vol_price_data, reshape_scaled_amount_price_data), axis=1)
        else:
            scaled_concat_price_data = np.concatenate(
                (reshape_scaled_high_price_data, reshape_scaled_open_price_data, reshape_scaled_close_price_data,
                 reshape_scaled_low_price_data), axis=1)
    elif model_configs.pred_price_type == 'low':
        scaled_target_price_data = scaled_low_price_data
        target_dataset_price_min_value = dataset_low_price_min_value
        target_dataset_price_max_value = dataset_low_price_max_value
        if model_configs.use_vol_amount:
            scaled_concat_price_data = np.concatenate(
                (reshape_scaled_low_price_data, reshape_scaled_open_price_data, reshape_scaled_high_price_data,
                 reshape_scaled_close_price_data,
                 reshape_scaled_vol_price_data, reshape_scaled_amount_price_data), axis=1)
        else:
            scaled_concat_price_data = np.concatenate(
                (reshape_scaled_low_price_data, reshape_scaled_open_price_data, reshape_scaled_high_price_data,
                 reshape_scaled_close_price_data), axis=1)
    elif model_configs.pred_price_type == 'open':
        scaled_target_price_data = scaled_open_price_data
        target_dataset_price_min_value = dataset_open_price_min_value
        target_dataset_price_max_value = dataset_open_price_max_value
        if model_configs.use_vol_amount:
            scaled_concat_price_data = np.concatenate(
                (reshape_scaled_open_price_data, reshape_scaled_close_price_data, reshape_scaled_high_price_data,
                 reshape_scaled_low_price_data,
                 reshape_scaled_vol_price_data, reshape_scaled_amount_price_data), axis=1)
        else:
            scaled_concat_price_data = np.concatenate(
                (reshape_scaled_open_price_data, reshape_scaled_close_price_data, reshape_scaled_high_price_data,
                 reshape_scaled_low_price_data), axis=1)
    else:
        raise NotImplementedError

    assert len(scaled_target_price_data) == len(dataset)
    assert len(scaled_target_price_data) == len(scaled_concat_price_data)
    all_years = [dataset[:, 0][i].year for i in range(len(dataset))]
    all_months = [dataset[:, 0][i].month for i in range(len(dataset))]
    all_days = [dataset[:, 0][i].day for i in range(len(dataset))]
    # 输出这一天是周中的第几天，Monday=0, Sunday=6
    all_weekdays = [dataset[:, 0][i].dayofweek for i in range(len(dataset))]
    # 判断是否在节日日期前后，生成特征: 1, 2, 0
    all_festival_periods_types = [create_is_holiday_periods_feature(all_years[i], all_months[i], all_days[i], periods=model_configs.festival_periods) for i in range(len(dataset))]

    history_prices = scaled_concat_price_data

    # [time_seq_length, num_prices_used]
    history_prices_for_test = history_prices[-model_configs.time_seq_length:]
    test_year_inputs = all_years[-model_configs.time_seq_length:]
    test_month_inputs = all_months[-model_configs.time_seq_length:]
    test_day_inputs = all_days[-model_configs.time_seq_length:]
    test_weekday_inputs = all_weekdays[-model_configs.time_seq_length:]
    test_festival_periods_type_inputs = all_festival_periods_types[-model_configs.time_seq_length:]

    last_year = all_years[-1]
    last_month = all_months[-1]
    last_day = all_days[-1]
    test_dec_year_inputs, test_dec_month_inputs, test_dec_day_inputs, test_dec_weekday_inputs = create_future_dates(last_year, last_month, last_day,
                                                                                                future_num_days=model_configs.max_pred_days)
    test_dec_festival_periods_type_inputs = [create_is_holiday_periods_feature(test_dec_year_inputs[i], test_dec_month_inputs[i], test_dec_day_inputs[i], periods=model_configs.festival_periods) for i in range(len(test_dec_day_inputs))]

    history_prices_for_test = np.reshape(np.asarray(history_prices_for_test, dtype=np.float32), [1, model_configs.time_seq_length, -1])
    test_year_inputs = np.reshape(np.asarray(test_year_inputs, dtype=np.int32), [1, model_configs.time_seq_length])
    test_month_inputs = np.reshape(np.asarray(test_month_inputs, dtype=np.int32), [1, model_configs.time_seq_length])-1
    test_day_inputs = np.reshape(np.asarray(test_day_inputs, dtype=np.int32), [1, model_configs.time_seq_length])-1
    test_weekday_inputs = np.reshape(np.asarray(test_weekday_inputs, dtype=np.int32), [1, model_configs.time_seq_length])
    test_festival_periods_type_inputs = np.reshape(np.asarray(test_festival_periods_type_inputs, dtype=np.int32),
                                                   [1, model_configs.time_seq_length])
    test_dec_year_inputs = np.reshape(np.asarray(test_dec_year_inputs, dtype=np.int32), [1, model_configs.max_pred_days])
    test_dec_month_inputs = np.reshape(np.asarray(test_dec_month_inputs, dtype=np.int32), [1, model_configs.max_pred_days])-1
    test_dec_day_inputs = np.reshape(np.asarray(test_dec_day_inputs, dtype=np.int32), [1, model_configs.max_pred_days])-1
    test_dec_weekday_inputs = np.reshape(np.asarray(test_dec_weekday_inputs, dtype=np.int32), [1, model_configs.max_pred_days])
    test_dec_festival_periods_type_inputs = np.reshape(np.asarray(test_dec_festival_periods_type_inputs, dtype=np.int32),
                                                       [1, model_configs.max_pred_days])

    return (history_prices_for_test, test_month_inputs, test_day_inputs, test_weekday_inputs, test_festival_periods_type_inputs, test_dec_month_inputs, test_dec_day_inputs, test_dec_weekday_inputs, test_dec_festival_periods_type_inputs, test_year_inputs, test_dec_year_inputs), \
            target_dataset_price_min_value, target_dataset_price_max_value


def data_iterator(model_configs, raw_data):
    (all_history_prices, all_month_inputs, all_day_inputs, all_weekday_inputs, all_festival_periods_types_inputs, all_dec_month_inputs, all_dec_day_inputs, all_dec_weekday_inputs, all_dec_festival_periods_types_inputs, all_targets, all_year_inputs, all_dec_year_inputs, all_targets_concat_prices) = raw_data
    data_len = len(all_targets)
    num_batches = data_len // model_configs.batch_size - 1

    for batch in range(num_batches):
        history_prices = np.zeros([model_configs.batch_size, model_configs.time_seq_length, model_configs.num_price_types_used], dtype=np.float32)
        year_inputs = np.zeros([model_configs.batch_size, model_configs.time_seq_length], dtype=np.int32)
        month_inputs = np.zeros([model_configs.batch_size, model_configs.time_seq_length], dtype=np.int32)
        day_inputs = np.zeros([model_configs.batch_size, model_configs.time_seq_length], dtype=np.int32)
        weekday_inputs = np.zeros([model_configs.batch_size, model_configs.time_seq_length], dtype=np.int32)
        festival_periods_types_inputs = np.zeros([model_configs.batch_size, model_configs.time_seq_length], dtype=np.int32)
        dec_year_inputs = np.zeros([model_configs.batch_size, model_configs.max_pred_days], dtype=np.int32)
        dec_month_inputs = np.zeros([model_configs.batch_size, model_configs.max_pred_days], dtype=np.int32)
        dec_day_inputs = np.zeros([model_configs.batch_size, model_configs.max_pred_days], dtype=np.int32)
        dec_weekday_inputs = np.zeros([model_configs.batch_size, model_configs.max_pred_days], dtype=np.int32)
        dec_festival_periods_types_inputs = np.zeros([model_configs.batch_size, model_configs.max_pred_days], dtype=np.int32)
        targets = np.zeros([model_configs.batch_size, model_configs.max_pred_days], dtype=np.float32)
        targets_concat_prices = np.zeros(
            [model_configs.batch_size, model_configs.max_pred_days, model_configs.num_price_types_used], dtype=np.float32)

        for i in range(model_configs.batch_size):
            data_index = batch * model_configs.batch_size + i
            history_price = all_history_prices[data_index]
            year_input = all_year_inputs[data_index]
            month_input = all_month_inputs[data_index]
            day_input = all_day_inputs[data_index]
            weekday_input = all_weekday_inputs[data_index]
            festival_periods_types_input = all_festival_periods_types_inputs[data_index]

            dec_year_input = all_dec_year_inputs[data_index]
            dec_month_input = all_dec_month_inputs[data_index]
            dec_day_input = all_dec_day_inputs[data_index]
            dec_weekday_input = all_dec_weekday_inputs[data_index]
            dec_festival_periods_types_input = all_dec_festival_periods_types_inputs[data_index]
            target = all_targets[data_index]
            target_concat_price = all_targets_concat_prices[data_index]

            history_prices[i] = history_price
            year_inputs[i] = year_input
            month_inputs[i] = month_input
            day_inputs[i] = day_input
            weekday_inputs[i] = weekday_input
            festival_periods_types_inputs[i] = festival_periods_types_input
            dec_year_inputs[i] = dec_year_input
            dec_month_inputs[i] = dec_month_input
            dec_day_inputs[i] = dec_day_input
            dec_weekday_inputs[i] = dec_weekday_input
            dec_festival_periods_types_inputs[i] = dec_festival_periods_types_input
            targets[i] = target
            targets_concat_prices[i] = target_concat_price

        yield (history_prices, month_inputs-1, day_inputs-1, weekday_inputs, festival_periods_types_inputs, dec_month_inputs-1, dec_day_inputs-1, dec_weekday_inputs, dec_festival_periods_types_inputs, targets, year_inputs, dec_year_inputs, targets_concat_prices)


if __name__ == "__main__":
    pass
