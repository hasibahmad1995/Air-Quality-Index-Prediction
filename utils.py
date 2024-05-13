import torch
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import pandas as pd
import numpy as np
import sys
import math


def get_networks(args, input_channel, output_channel):
    if args.model == 'tcn':
        from models.tcn import TCN
        net = TCN(input_channel, output_channel, [args.nhid] * args.levels, args.ksize, args.dropout, args.horizon)
    elif args.model == 'gru':
        from models.rnn import GRU
        net = GRU(input_channel, args.nhid, output_channel, args.levels, args.dropout, args.horizon)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.cuda:
        net = net.cuda()
    return net


def generate_polution_data(mode, path='./data/polution.csv'):
    df = pd.read_csv(path)

    # One-hot encode feature'cbwd'
    temp = pd.get_dummies(df['cbwd'], prefix='cbwd')
    df = pd.concat([df, temp], axis=1)
    del df['cbwd'], temp,

    # split the data set into three parts: the training set (60%) the validation set (20%), and the test set (20%).
    train, test, valid = np.split(df, [int(.6 * len(df)), int(.8 * len(df))])
    x_train = train.loc[:, ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE',
                            'cbwd_cv']].values.copy()
    x_test = test.loc[:, ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE',
                          'cbwd_cv']].values.copy()
    x_valid = valid.loc[:, ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE',
                            'cbwd_cv']].values.copy()
    y_train = train.loc[:, 'pm2.5'].values.copy()
    y_test = test.loc[:, 'pm2.5'].values.copy()
    y_valid = valid.loc[:, 'pm2.5'].values.copy()

    x = np.concatenate((x_train, x_test, x_valid), axis=0)
    y = np.concatenate((y_train, y_test, y_valid), axis=0)

    # Normalization
    # z-score transform x (the input feature)
    for i in range(x_train.shape[1] - 4):
        temp_mean = np.nanmean(x_train[:, i])  # mean
        temp_std = np.nanstd(x_train[:, i])  # variance
        x_train[:, i] = (x_train[:, i] - temp_mean) / temp_std
        x_test[:, i] = (x_test[:, i] - temp_mean) / temp_std
        x_valid[:, i] = (x_valid[:, i] - temp_mean) / temp_std

    # z-score transform y (the output value)
    y_mean = np.nanmean(y_train)
    y_std = np.nanstd(y_train)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    y_valid = (y_valid - y_mean) / y_std

    print(y_mean, y_std)

    if mode == 'train':
        return x_train, y_train, x_valid, y_valid, x_test, y_test, y_mean, y_std
    elif mode == 'test':
        return x_train, y_train, x_valid, y_valid, x_test, y_test, y_mean, y_std


def generate_samples(x_data, y_data, seq_length, horizon=1):
    if x_data.dtype == np.object_ or np.isnan(x_data).any():
        x_data = np.nan_to_num(x_data).astype(np.float32)
    if y_data.dtype == np.object_ or np.isnan(y_data).any():
        y_data = np.nan_to_num(y_data).astype(np.float32)
    
    m = x_data.shape[0]
    # Initialize the output tensors.
    x = torch.zeros([m - seq_length - (horizon - 1), x_data.shape[1], seq_length])
    y = torch.zeros([m - seq_length - (horizon - 1), y_data.shape[1], seq_length])
    
    # Convert numpy arrays to tensors.
    x_train = torch.from_numpy(x_data)
    y_train = torch.from_numpy(y_data)

    idx = 0
    for i in range(x.shape[0]):
        if torch.isnan(x_train[i:i + seq_length]).any() or torch.isnan(y_train[i + horizon:i + seq_length + horizon]).any():
            continue

        x[idx, :, :] = x_train[i:i + seq_length].transpose(1, 0)
        y[idx, :, :] = y_train[i + horizon:i + seq_length + horizon].transpose(1, 0)
        idx += 1

    print(f"Valid samples count: {idx}")
    return Variable(x[:idx]), Variable(y[:idx])


def output_restore(output, y_gap, y_min):
    if output.shape[1] == 1:
        output = output * y_gap + y_min
    else:
        for i in range(len(y_gap)):
            output[:, i] = output[:, i] * y_gap[i] + y_min[i]
    return output


def compute_MSE(output, actual, path="train", eps=1e-5, horizon=1):
    MAPE, MSE = 0, 0
    idx = 0
    mse_loss = []
    cnt = 0
    idx1 = 0

    for i in range(0, output.shape[0]):
        idx += 1
        out = output[i:i + 1, :, 0: horizon]
        act = actual[i:i + 1, :, 0: horizon]
        mse = np.mean(np.power(out - act, 2))
        MSE += mse

        for j in range(horizon):
            o1 = output[i:i + 1, :, j:j + 1]
            a1 = actual[i:i + 1, :, j:j + 1]
            idx1 += 1
            if interval(o1) == interval(a1):
                cnt += 1

    return MSE / idx, cnt / idx1


def interval(x):
    if x < 35:
        return 1
    elif x < 75:
        return 2
    elif x < 115:
        return 3
    elif x < 150:
        return 4
    elif x < 250:
        return 5
    elif x < 350:
        return 6
    elif x < 500:
        return 7
    else:
        return 8


def compute_accuracy(out, act, horizon=1):
    cnt = np.zeros((20, 1))
    mse = np.zeros((20, 1))
    total = 0
    level1 = np.zeros((20, 1))
    level6 = np.zeros((20, 1))
    num1 = np.zeros((20, 1))
    num6 = np.zeros((20, 1))

    mse1 = np.zeros((20, 1))
    mse6 = np.zeros((20, 1))

    print(out.shape, act.shape)
    for i in range(act.shape[0]):
        total += 1
        num1[interval(np.round(act[i:i + 1, :, 0:1]))] += 1
        num6[interval(np.round(act[i:i + 1, :, 5:6]))] += 1

        mse1[interval(np.round(act[i:i + 1, :, 0:1]))] += (out[i, :, 0] - act[i, :, 0]) ** 2
        mse6[interval(np.round(act[i:i + 1, :, 5:6]))] += (out[i, :, 5] - act[i, :, 5]) ** 2

        if interval(np.round(out[i:i + 1, :, 0:1])) == interval(np.round(act[i:i + 1, :, 0:1])):
            level1[interval(np.round(out[i:i + 1, :, 0:1]))] += 1
        if interval(np.round(out[i:i + 1, :, 5:6])) == interval(np.round(act[i:i + 1, :, 5:6])):
            level6[interval(np.round(out[i:i + 1, :, 5:6]))] += 1

        for j in range(horizon):
            mse[j] += (out[i, :, j] - act[i, :, j]) ** 2
            if interval(np.round(out[i:i + 1, :, j:j + 1])) == interval(np.round(act[i:i + 1, :, j:j + 1])):
                cnt[j] += 1
    for i in range(horizon):
        print(i, np.round(mse[i] / total, 4), np.round(cnt[i] / total, 4), "\t", end="")
    print("")
    print("======================================")
    print("hour1:")
    for i in range(1, 8):
        print(np.round(level1[i] / num1[i], 4), "\t", end="")
    print("")
    for i in range(1, 8):
        print(np.round(mse1[i] / num1[i], 2), "\t", end="")
    print("")
    print("=======================================")
    print("hour6:")
    for i in range(1, 8):
        print(np.round(level6[i] / num6[i], 4), "\t", end="")
    print("")
    for i in range(1, 8):
        print(np.round(mse6[i] / num6[i], 2), "\t", end="")
    print("")
