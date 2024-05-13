import argparse
import torch.nn as nn
import torch.optim as optim
from ast import literal_eval
from utils import *
import matplotlib.pyplot as plt
import os
from global_settings import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def compute(out, act, horizon=1):
    cnt = np.zeros((20, 1))
    mse = np.zeros((20, 1))
    total = 0
    level1 = np.zeros((20, 1))
    level3 = np.zeros((20, 1))
    level6 = np.zeros((20, 1))
    num_a1 = np.zeros((20, 1))
    num_a3 = np.zeros((20, 1))
    num_a6 = np.zeros((20, 1))
    num_o1 = np.zeros((20, 1))
    num_o3 = np.zeros((20, 1))
    num_o6 = np.zeros((20, 1))

    mse1 = np.zeros((20, 1))
    mse3 = np.zeros((20, 1))
    mse6 = np.zeros((20, 1))

    print(out.shape, act.shape)
    for i in range(act.shape[0]):
        total += 1
        num_a1[interval(np.round(act[i:i + 1, :, 0:1]))] += 1
        num_a3[interval(np.round(act[i:i + 1, :, 2:3]))] += 1
        num_a6[interval(np.round(act[i:i + 1, :, 5:6]))] += 1

        num_o1[interval(np.round(out[i:i + 1, :, 0:1]))] += 1
        num_o3[interval(np.round(out[i:i + 1, :, 2:3]))] += 1
        num_o6[interval(np.round(out[i:i + 1, :, 5:6]))] += 1

        mse1[interval(np.round(act[i:i + 1, :, 0:1]))] += (out[i, :, 0] - act[i, :, 0]) ** 2
        mse3[interval(np.round(act[i:i + 1, :, 2:3]))] += (out[i, :, 2] - act[i, :, 2]) ** 2
        mse6[interval(np.round(act[i:i + 1, :, 5:6]))] += (out[i, :, 5] - act[i, :, 5]) ** 2

        if interval(np.round(out[i:i + 1, :, 0:1])) == interval(np.round(act[i:i + 1, :, 0:1])):
            level1[interval(np.round(out[i:i + 1, :, 0:1]))] += 1
        if interval(np.round(out[i:i + 1, :, 2:3])) == interval(np.round(act[i:i + 1, :, 2:3])):
            level3[interval(np.round(out[i:i + 1, :, 2:3]))] += 1
        if interval(np.round(out[i:i + 1, :, 5:6])) == interval(np.round(act[i:i + 1, :, 5:6])):
            level6[interval(np.round(out[i:i + 1, :, 5:6]))] += 1

        for j in range(horizon):
            mse[j] += (out[i, :, j] - act[i, :, j]) ** 2
            if interval(np.round(out[i:i + 1, :, j:j + 1])) == interval(np.round(act[i:i + 1, :, j:j + 1])):
                cnt[j] += 1

    col1 = []
    col_total = []
    for i in range(0, 8):
        col1.append([])

    for i in range(0,horizon):
        col_total.append([])

    for i in range(horizon):
        col_total[i].append(np.round(cnt[i][0] / total, 4))
        col_total[i].append(np.round(mse[i][0] / total, 4))
        print(i, np.round(mse[i] / total, 4), np.round(cnt[i] / total, 4), "\t", end="")
    print("")
    print("======================================")
    print("hour1:")
    for i in range(1, 8):
        p = np.round(level1[i][0] / num_o1[i][0],4)
        r = np.round(level1[i][0] / num_a1[i][0],4)
        col1[i].append(r)
        col1[i].append(p)
        col1[i].append(np.round(2 * p * r / (p + r),4))
        print(np.round(level1[i] / num_a1[i], 4), "\t", end="")
    print("")
    for i in range(1, 8):
        col1[i].append(np.round(mse1[i][0] / num_a1[i][0], 2))
        col1[i].append("")
        print(np.round(mse1[i][0] / num_a1[i][0], 2), "\t", end="")
    print("")
    print("=======================================")

    print("hour3:")
    for i in range(1, 8):
        p = np.round(level3[i][0] / num_o3[i][0], 4)
        r = np.round(level3[i][0] / num_a3[i][0], 4)
        col1[i].append(r)
        col1[i].append(p)
        col1[i].append(np.round(2 * p * r / (p + r),4))
        print(np.round(level3[i] / num_a3[i], 4), "\t", end="")
    print("")
    for i in range(1, 8):
        col1[i].append(np.round(mse3[i][0] / num_a3[i][0], 2))
        col1[i].append("")
        print(np.round(mse3[i] / num_a3[i], 2), "\t", end="")
    print("")
    print("=======================================")

    print("hour6:")
    for i in range(1, 8):
        p = np.round(level6[i][0] / num_o6[i][0], 4)
        r = np.round(level6[i][0] / num_a6[i][0], 4)
        col1[i].append(r)
        col1[i].append(p)
        col1[i].append(np.round(2 * p * r / (p + r),4))
        print(np.round(level6[i] / num_a6[i], 4), "\t", end="")
    print("")
    for i in range(1, 8):
        col1[i].append(np.round(mse6[i][0] / num_a6[i][0], 2))
        print(np.round(mse6[i] / num_a6[i], 2), "\t", end="")
    print("")

    return col1,col_total


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=128, help='batch size')
    parser.add_argument('-cuda', type=literal_eval, default=False, help='use CUDA')
    parser.add_argument('-dropout', type=float, default=0.3, help='dropout applied to layers')
    parser.add_argument('-clip', type=float, default=0.15, help='gradient clip, -1 means no clip')
    parser.add_argument('-ksize', type=int, default=3, help='kernel size')
    parser.add_argument('-levels', type=int, default=8, help='# of levels')
    parser.add_argument('-seq_len', type=int, default=48, help='sequence length')
    parser.add_argument('-nhid', type=int, default=40, help='number of hidden units per layer ')
    parser.add_argument('-seed', type=int, default=1111, help='random seed')
    parser.add_argument('-horizon', type=int, default=6, help='prediction time steps')
    parser.add_argument('-model', type=str, default='tcn', help='prediction model')
    parser.add_argument('-data', type=str, default='pollution', help='dataset')
    parser.add_argument('-loss', type=str, default='square', help='dataset')

    args = parser.parse_args()

    torch.manual_seed(args.seed)    

    batch_size = args.batch_size
    seq_length = args.seq_len
    horizon = args.horizon
    channel_sizes = [args.nhid] * args.levels

    if args.data == 'pollution':
        x_train, y_train, x_valid, y_valid, x_test, y_test, y_min, y_gap = generate_polution_data('test')

    print(f'Test: {x_test.shape}')

    if len(y_test.shape) == 1:
        y_train = np.expand_dims(y_train, axis=1)
        y_valid = np.expand_dims(y_valid, axis=1)
        y_test = np.expand_dims(y_test, axis=1)


    input_channels = x_test.shape[1]
    n_classes = y_test.shape[1]

    train_x, train_y = generate_samples(x_train, y_train, seq_length, horizon)
    valid_x, valid_y = generate_samples(x_valid, y_valid, seq_length, horizon)
    test_x, test_y = generate_samples(x_test, y_test, seq_length, horizon)   


    df = pd.read_excel("index_num.xlsx")
    idx = df.copy().values
    test_idx = idx[train_x.shape[0]+valid_x.shape[0]:]
    print("Length of the test dataset: ",len(test_idx))

    data_x = torch.cat([train_x, valid_x], 0)
    data_x = torch.cat([data_x, test_x], 0)
    data_y = torch.cat([train_y, valid_y], 0)
    data_y = torch.cat([data_y, test_y], 0)

    if args.cuda:
        test_x, test_y = test_x.cuda(), test_y.cuda()
        train_x, train_y = train_x.cuda(), train_y.cuda()
        valid_x, valid_y = valid_x.cuda(), valid_y.cuda()
        data_x, data_y = data_x.cuda(), data_y.cuda()

    
    y_train = output_restore(y_train, y_gap, y_min)
    y_valid = output_restore(y_valid, y_gap, y_min)
    y_test = output_restore(y_test, y_gap, y_min)

    model = get_networks(args, input_channels, n_classes)
    checkpoint_path = os.path.join(CHECKPOINT_PATH, args.data, '{model}-horizon{horizon}-{loss}.pkl')
    # print(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path.format(model=args.model, horizon=str(horizon), loss=args.loss)))

    # test
    model.eval()
    num_test = test_x.size()[0]
    test_output = torch.Tensor()
    test_actual = torch.Tensor()
    if args.cuda:
        test_output = test_output.cuda()
        test_actual = test_actual.cuda()
    batch_idx = 1
    with torch.no_grad():
        for k in range(0, num_test, batch_size):
            for i in range(k, k + batch_size):
                if i >= test_x.shape[0]:
                    break
                idx = test_idx[i]
                if i == k:
                    x = data_x[idx]
                    y = data_y[idx]
                else:
                    x = torch.cat([x, data_x[idx]], 0)
                    y = torch.cat([y, data_y[idx]], 0)
            output = model(x)  # (N, output_size * horizon)
            output = output.reshape(output.size()[0], n_classes, horizon)
            test_output = torch.cat([test_output, output], 0)
            test_actual = torch.cat([test_actual, y[:, :, -horizon:]], 0)
            batch_idx += 1
    if args.cuda:
        test_output = test_output.cpu().detach().numpy()
        test_actual = test_actual.cpu().detach().numpy()
    else:
        test_output = test_output.detach().numpy()
        test_actual = test_actual.detach().numpy()

    true_output = output_restore(test_output, y_gap, y_min)
    true_actual = output_restore(test_actual, y_gap, y_min)

    x = []
    for i in range(1,501):
        x.append(i)

    plt.plot(x, np.squeeze(true_actual[500:1000,:,0:1]), label="actual")
    plt.plot(x, np.squeeze(true_output[500:1000,:,0:1]), label="GRU-IEEM")
    plt.xlabel("Time(h)",fontsize=14)
    plt.ylabel(r"PM2.5(ug/$m^3$)",fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig("gru-horizon1-IEEM1.pdf", bbox_inches="tight")
    plt.show()

    result_path = os.path.join(RESULT_PATH, args.data, args.loss, args.model, 'horizon' + str(horizon))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    result_path = os.path.join(result_path, f'test-result.xlsx')
    # print(result_path)
    MSE, Accuracy = compute_MSE(true_output, true_actual,horizon=horizon)
    # compute_accuracy(true_output, true_actual, horizon=horizon)
    print(f'Test set: MSE = {MSE}, Accuracy = {Accuracy} with predicted horizon: {args.horizon}')

#     col,col_total = compute(true_output, true_actual, horizon=horizon)
    title = ["hour1","","","","","hour3","","","","","hour6","","",""]
    title2 = ["Recall","Precision","F1-score","MSE","","Recall","Precision","F1-score","MSE","","Recall","Precision","F1-score","MSE"]

    # save result
#     sheet1 = pd.DataFrame({
#         "time": title,
#         "IEEM loss function": title2,
#         '0-35': col[1],
#         '35-75': col[2],
#         '75-115': col[3],
#         '115-150': col[4],
#         '150-250': col[5],
#         '250-350': col[6],
#         '350-500': col[7]
#     })

#     title1 = ["accuracy", "MSE"]
#     sheet2 = pd.DataFrame({
#         f"IEEM loss function": title1,
#         'hour1': col_total[0],
#         'hour2': col_total[1],
#         'hour3': col_total[2],
#         'hour4': col_total[3],
#         'hour5': col_total[4],
#         'hour6': col_total[5]
#     })

    sheet3 = pd.DataFrame({
        'test_output': true_output[:, 0].tolist(),
        'test_actual': true_actual[:, 0].tolist()
    })
    writer = pd.ExcelWriter(result_path)
#     sheet1.to_excel(writer, 'result', index=False)
#     sheet2.to_excel(writer, 'total', index=False)
    sheet3.to_excel(writer, 'test', index=False)
    writer._save()
    writer.close()
