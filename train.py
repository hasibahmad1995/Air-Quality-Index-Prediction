import argparse
import torch.nn as nn
import torch.optim as optim
from ast import literal_eval
from models.ieem import IEEM
from utils import *
import os
from global_settings import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train(epoch):
    model.train()
    num_train = train_x.size()[0]
    total_loss = 0
    temp_loss = 0
    batch_idx = 1

    train_output = torch.Tensor()
    train_actual = torch.Tensor()
    if args.cuda:
        train_output = train_output.cuda()
        train_actual = train_actual.cuda()

    for k in range(0, num_train, batch_size):
        for i in range(k, k + batch_size):
            if i >= train_idx.shape[0]:
                break
            idx = train_idx[i]
            if i == k:
                x = data_x[idx]
                y = data_y[idx]
            else:
                x = torch.cat([x, data_x[idx]], 0)
                y = torch.cat([y, data_y[idx]], 0)

        optimizer.zero_grad()
        output = model(x)  # (N, output_size * horizon)
        output = output.reshape(output.size()[0], n_classes, horizon)

        # p-step ahead loss
        loss = loss_function(output, y[:, :, -horizon:])

        train_output = torch.cat([train_output, output], 0)  # 取多点作为预测结果
        train_actual = torch.cat([train_actual, y[:, :, -horizon:]], 0)

        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)  # 梯度裁剪
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()
        temp_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            cur_loss = temp_loss / (args.log_interval * batch_size)
            processed = min(k + batch_size, num_train)
            # print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
            #     epoch, processed, num_train, 100. * processed / num_train, lr, cur_loss))
            temp_loss = 0

    if args.cuda:
        train_output = train_output.cpu().detach().numpy()
        train_actual = train_actual.cpu().detach().numpy()
    else:
        train_output = train_output.detach().numpy()
        train_actual = train_actual.detach().numpy()

    true_output = output_restore(train_output, y_std, y_mean)
    true_actual = output_restore(train_actual, y_std, y_mean)

    MSE, Accuracy = compute_MSE(true_output, true_actual, horizon=horizon)
    # compute_accuracy(true_output, true_actual, horizon=horizon)
    print(
        f'Epoch: {epoch} Train set: MSE = {MSE}, Accuracy = {Accuracy} with predicted horizon: {horizon}')

    result_loss = MSE
    return total_loss / num_train, result_loss, true_output, true_actual


def eval_training(epoch):
    model.eval()
    num_val = valid_x.size()[0]
    batch_idx = 1
    total_loss = 0
    val_output = torch.Tensor()
    val_actual = torch.Tensor()
    if args.cuda:
        val_output = val_output.cuda()
        val_actual = val_actual.cuda()
    with torch.no_grad():
        for k in range(0, num_val, batch_size):
            for i in range(k, k + batch_size):
                if i >= valid_idx.shape[0]:
                    break
                idx = valid_idx[i]
                if i == k:
                    x = data_x[idx]
                    y = data_y[idx]
                else:
                    x = torch.cat([x, data_x[idx]], 0)
                    y = torch.cat([y, data_y[idx]], 0)

            output = model(x)  # (N, output_size * horizon)
            output = output.reshape(output.size()[0], n_classes, horizon)

            # p-step ahead loss
            loss = loss_function(output, y[:, :, -horizon:])
            val_output = torch.cat([val_output, output], 0)  # 取单点作为预测结果
            val_actual = torch.cat([val_actual, y[:, :, -horizon:]], 0)

            batch_idx += 1
            total_loss += loss.item()
    if args.cuda:
        val_output = val_output.cpu().detach().numpy()
        val_actual = val_actual.cpu().detach().numpy()
    else:
        val_output = val_output.detach().numpy()
        val_actual = val_actual.detach().numpy()
    true_output = output_restore(val_output, y_std, y_mean)
    val_actual = output_restore(val_actual, y_std, y_mean)
    MSE, Accuracy = compute_MSE(true_output, val_actual, horizon=horizon)
    # compute_accuracy(true_output, val_actual, horizon=horizon)
    print(
        f'Epoch: {epoch} Val set: MSE = {MSE}, Accuracy = {Accuracy} with predicted horizon: {horizon}')

    result_loss = MSE

    return total_loss / num_val, result_loss, true_output, val_actual


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=128, help='batch size')
    parser.add_argument('-cuda', type=literal_eval, default=False, help='use CUDA')
    parser.add_argument('-dropout', type=float, default=0.3, help='dropout applied to layers')
    parser.add_argument('-clip', type=float, default=0.15, help='gradient clip, -1 means no clip')
    parser.add_argument('-epochs', type=int, default=5, help='upper epoch limit')
    parser.add_argument('-ksize', type=int, default=3, help='kernel size')
    parser.add_argument('-horizon', type=int, default=6, help='kernel size')
    parser.add_argument('-levels', type=int, default=8, help='# of levels')
    parser.add_argument('-seq_len', type=int, default=48, help='sequence length')
    parser.add_argument('-log_interval', type=int, default=100, metavar='N', help='report interval')
    parser.add_argument('-lr', type=float, default=4e-3, help='initial learning rate')
    parser.add_argument('-optim', type=str, default='Adam', help='optimizer to use (default: Adam)')
    parser.add_argument('-nhid', type=int, default=40, help='number of hidden units per layer ')
    parser.add_argument('-seed', type=int, default=1111, help='random seed')
    parser.add_argument('-model', type=str, default='gru', help='prediction model')
    parser.add_argument('-data', type=str, default='pollution', help='dataset')
    parser.add_argument('-loss', type=str, default='square', help='IEEM,square')
    args = parser.parse_args()

    torch.manual_seed(args.seed)  # set the random seed
    print(f'Using {args.model} to train.')
    print(args)

    batch_size = args.batch_size
    seq_length = args.seq_len
    epochs = args.epochs
    lr = args.lr
    channel_sizes = [args.nhid] * args.levels
    horizon = args.horizon

    if args.data == 'pollution':
        x_train, y_train, x_valid, y_valid, x_test, y_test, y_mean, y_std = generate_polution_data('train')

    # expand y dimension
    if len(y_train.shape) == 1:
        y_train = np.expand_dims(y_train, axis=1)
        y_valid = np.expand_dims(y_valid, axis=1)
        y_test = np.expand_dims(y_test, axis=1)

    # input and output dimension
    input_channels = x_train.shape[1]
    n_classes = y_train.shape[1]

    print(f'Train: {x_train.shape}, Valid: {x_valid.shape}')

    train_x, train_y = generate_samples(x_train, y_train, seq_length, horizon)
    valid_x, valid_y = generate_samples(x_valid, y_valid, seq_length, horizon)
    test_x, test_y = generate_samples(x_test, y_test, seq_length, horizon)

    print(train_x.shape, train_y.shape)
    print(valid_x.shape, valid_y.shape)
    print(test_x.shape, test_y.shape)
    print("----------------------------")
    # return the original value
    y_train = output_restore(y_train, y_std, y_mean)
    y_valid = output_restore(y_valid, y_std, y_mean)
    y_test = output_restore(y_test, y_std, y_mean)

    # shuffle the data set using the random index
    df = pd.read_excel("index_num.xlsx")
    idx = df.copy().values
    train_idx = idx[0:train_x.shape[0]]
    valid_idx = idx[train_x.shape[0]:train_x.shape[0] + valid_x.shape[0]]
    print(len(valid_idx))

    data_x = torch.cat([train_x, valid_x], 0)
    data_x = torch.cat([data_x, test_x], 0)
    data_y = torch.cat([train_y, valid_y], 0)
    data_y = torch.cat([data_y, test_y], 0)

    model = get_networks(args, input_channels, n_classes)
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

    if args.loss == "IEEM":
        loss_function = IEEM()
    else:
        loss_function = nn.MSELoss()
    print(args.loss)
    # loss_function = nn.L1Loss()
    # loss_function = nn.SmoothL1Loss()
    

    if args.cuda:
        train_x, train_y = train_x.cuda(), train_y.cuda()
        valid_x, valid_y = valid_x.cuda(), valid_y.cuda()
        data_x, data_y = data_x.cuda(), data_y.cuda()

    checkpoint_path = os.path.join(CHECKPOINT_PATH, args.data)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{model}-horizon{horizon}-{loss}.pkl')

    train_loss_list = []
    val_loss_list = []
    best_MSE = float('inf')
    best_epoch = 1

    sheet1 = pd.DataFrame()
    sheet2 = pd.DataFrame()

    for epoch in range(1, epochs+1):
        train_loss, train_MSE, train_output, train_actual = train(epoch)
        val_loss, val_MSE, val_output, valid_actual = eval_training(epoch)
        train_loss_list.append(train_MSE)
        val_loss_list.append(val_MSE)

        if best_MSE > val_MSE:
            best_MSE = val_MSE
            best_epoch = epoch
            train_output_best = train_output
            valid_output_best = val_output

            train_actual_best = train_actual
            valid_actual_best = valid_actual
            torch.save(model.state_dict(), checkpoint_path.format(model=args.model, horizon=str(horizon), loss=args.loss))

    print(f'Best val epoch: {best_epoch}')

    sheet1 = pd.DataFrame({
        'training output value': train_output_best[:, 0].tolist(),
        'training actual value': train_actual_best[:, 0].tolist()
    })
    sheet2 = pd.DataFrame({
        'validation output value': valid_output_best[:, 0].tolist(),
        'validation actual value': valid_actual_best[:, 0].tolist()
    })
    sheet4 = pd.DataFrame({
        'training loss': train_loss_list,
        'validation loss': val_loss_list
    })

    result_path = os.path.join(RESULT_PATH, args.data, args.model, 'horizon' + str(horizon))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = os.path.join(result_path, f'train_valid_result-{args.loss}.xlsx')

    writer = pd.ExcelWriter(result_path)
    sheet1.to_excel(writer, 'train', index=False)
    sheet2.to_excel(writer, 'val', index=False)
    sheet4.to_excel(writer, 'loss', index=False)
    writer._save()
    writer.close()
