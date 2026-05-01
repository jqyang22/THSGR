import torch
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from sklearn.metrics import classification_report,cohen_kappa_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
import os
import json
from datetime import datetime
import time
from models.THSGR import THSGR
from loadData import data_pipe
import pandas as pd
from loadData import data_reader
from thop import profile
parser = argparse.ArgumentParser(description='THSGR')
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--path-config', type=str, default='/THSGR/config/config.yaml')
parser.add_argument('--save-name-pre', default='', type=str, metavar='PATH', help='path to cache (default: none)')
parser.add_argument('--print-config', action='store_true', default=False)
parser.add_argument('--print-data-info', action='store_true', default=False)
parser.add_argument('--plot-loss-curve', action='store_true', default=False)
parser.add_argument('--show-results', action='store_true', default=True)
parser.add_argument('--save-results', action='store_true', default=True)
args = parser.parse_args()
if args.save_name_pre == '':
    args.results_dir = datetime.now().strftime("%Y%m%d-%H%M")
config = yaml.load(open("/THSGR/config/config.yaml", "r"),
                   Loader=yaml.FullLoader)
dataset_name = config["data_input"]["dataset_name"]
classes = config["data_input"]["classes"]
patch_size = config["data_input"]["patch_size"]
num_components = config["data_transforms"]["num_components"]
batch_size = config["data_transforms"]["batch_size"]
remove_zero_labels = config["data_transforms"]["remove_zero_labels"]
max_epoch = config["network_config"]["max_epoch"]
learning_rate = config["network_config"]["learning_rate"]
weight_decay = config["network_config"]["weight_decay"]
lb_smooth = config["network_config"]["lb_smooth"]
num_nodes = config["network_config"]["num_nodes"]
log_interval = config["result_output"]["log_interval"]
path_weight = config["result_output"]["path_weight"]
path_result = config["result_output"]["path_result"]
if not os.path.exists(path_result):
    os.mkdir(path_result)
if not os.path.exists(path_weight):
    os.mkdir(path_weight)
if dataset_name == 'Houston_2013':
    row = 349
    col = 1905
elif dataset_name == 'Augsburg':
    row = 332
    col = 485
elif dataset_name == 'Berlin':
    row = 1723
    col = 476
def DrawResult(labels, dataset_name):
    labels -= 1
    num_class = labels.max() + 1
    if dataset_name == 'PaviaU':
        row = 610
        col = 340
        palette = np.array([[216, 191, 216],
                            [0, 255, 0],
                            [0, 255, 255],
                            [45, 138, 86],
                            [255, 0, 255],
                            [255, 165, 0],
                            [159, 31, 239],
                            [255, 0, 0],
                            [255, 255, 0]])
        palette = palette * 1.0 / 255
    elif dataset_name == 'IndianPines':
        row = 145
        col = 145
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240],
                            [255, 127, 80],
                            [127, 255, 212],
                            [218, 112, 214],
                            [160, 82, 45],
                            [127, 255, 0],
                            [216, 191, 216],
                            [238, 0, 0]])
        palette = palette * 1.0 / 255
    elif dataset_name == 'Houston_2013' or dataset_name == 'Augsburg' or dataset_name == 'Berlin':
        if dataset_name == 'Houston_2013':
            row = 349
            col = 1905
        elif dataset_name == 'Augsburg':
            row = 332
            col = 485
        elif dataset_name == 'Berlin':
            row = 1723
            col = 476
        palette = np.array([[0, 205, 0],
                            [127, 255, 0],
                            [46, 139, 87],
                            [0, 139, 0],
                            [160, 82, 45],
                            [0, 255, 255],
                            [255, 255, 255],
                            [216, 191, 216],
                            [255, 0, 0],
                            [139, 0, 0],
                            [0, 0, 0],
                            [255, 255, 0],
                            [238, 154, 0],
                            [85, 26, 139],
                            [255, 127, 80]])
        palette = palette * 1.0 / 255
    elif dataset_name == 'MUUFL':
        row = 325
        col = 220
        palette = np.array([[0, 205, 0],
                            [127, 255, 0],
                            [46, 139, 87],
                            [0, 139, 0],
                            [160, 82, 45],
                            [0, 255, 255],
                            [255, 255, 255],
                            [216, 191, 216],
                            [255, 0, 0],
                            [139, 0, 0],
                            [0, 0, 0]])
        palette = palette * 1.0 / 255
    elif dataset_name == 'Trento':
        row = 600
        col = 166
        palette = np.array([[0, 205, 0],
                            [127, 255, 0],
                            [46, 139, 87],
                            [0, 139, 0],
                            [160, 82, 45],
                            [0, 255, 255]])
        palette = palette * 1.0 / 255
    X_result = np.zeros((labels.shape[0], 3))
    for i in range(0, num_class):
        X_result[np.where(labels == i), 0] = palette[i, 0]
        X_result[np.where(labels == i), 1] = palette[i, 1]
        X_result[np.where(labels == i), 2] = palette[i, 2]
    X_result = np.reshape(X_result, (row, col, 3))
    return X_result
for w in range(9, 10, 2):
    w = w/10
    Experiment_num = 5
    Experiment_result = np.zeros([classes + 6, Experiment_num + 2])
    for count in range(0, Experiment_num):
        tic0 = time.time()
        train_loader, test_loader, train_label, test_label, pre_loader, data_gt, train_dataset = data_pipe.get_data(model_name="THSGR",
                                    path_config=args.path_config, print_config=args.print_config,
                                    print_data_info=args.print_data_info)
        net = THSGR(input_channels=num_components, num_nodes=(np.max(test_label)+1)*num_nodes, num_classes=np.max(test_label)+1, patch_size=patch_size).to(args.device)
        criterion = LabelSmoothingCrossEntropy(smoothing=lb_smooth)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer, milestones=[max_epoch // 2, (5 * max_epoch) // 6], gamma=0.1)
        for data, data_LiDAR, target in train_loader:
            flops, params = profile(net, (
            data.float().to(args.device), data_LiDAR.float().to(args.device), 0.9))
            print('FLOPs: {} G, PARAMS: {} M'.format(flops / 1e9,
                                                     params / 1e6))
        def train(net, max_epoch, criterion, optimizer, scheduler):
          best_loss = 9999
          train_losses = []
          net.train()
          for epoch in range(1, max_epoch+1):
            correct = 0
            for data, data_LiDAR, target in train_loader:
              data = data.to(args.device)
              data_LiDAR = data_LiDAR.to(args.device)
              target = target.to(args.device)
              optimizer.zero_grad()
              output = net(data, data_LiDAR, w)
              loss = criterion(output, target)
              loss.backward()
              optimizer.step()
              pred = output.data.max(1, keepdim=True)[1]
              correct += pred.eq(target.data.view_as(pred)).sum()
            scheduler.step()
            train_losses.append(loss.cpu().detach().item())
            if epoch % log_interval == 0:
              print('Train Epoch: {}\tLoss: {:.6f} \tAccuracy: {:.6f}'.format(epoch,  loss.item(),  correct / len(train_loader.dataset)))
            if loss.item() < best_loss:
              best_loss = loss.item()
              torch.save(net.state_dict(), path_weight + 'model.pth')
              torch.save(optimizer.state_dict(), path_weight + 'optimizer.pth')
          return train_losses
        tic1 = time.time()
        train_losses = train(net, max_epoch, criterion, optimizer, scheduler)
        toc1 = time.time()
        def test(net):
          net.eval()
          test_losses = []
          test_preds = []
          test_loss = 0
          correct = 0
          net.load_state_dict(torch.load(path_weight + 'model.pth'))
          with torch.no_grad():
            for data, data_LiDAR, target in test_loader:
              data = data.to(args.device)
              data_LiDAR = data_LiDAR.to(args.device)
              target = target.to(args.device)
              output = net(data, data_LiDAR, w)
              test_loss += criterion(output, target).item()
              test_pred = output.data.max(1, keepdim=True)[1]
              correct += test_pred.eq(target.data.view_as(test_pred)).sum()
              test_label = torch.argmax(output, dim=1)
              test_preds.append(test_label.cpu().numpy().tolist())
          test_losses.append(test_loss)
          print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} \
                ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
          return test_losses, test_preds
        tic2 = time.time()
        test_losses, test_preds = test(net)
        toc2 = time.time()
        if args.plot_loss_curve:
            fig = plt.figure()
            plt.plot(range(max_epoch), train_losses, color='blue')
            plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
            plt.xlabel('number of training examples seen')
            plt.ylabel('negative log likelihood loss')
            plt.show()
        y_pred_test = [j for i in test_preds for j in i]
        num_tes = np.zeros([classes])
        num_tes_pred = np.zeros([classes])
        y_pre = np.array(y_pred_test)
        y_tes = np.array(test_label)
        for k in y_tes:
            num_tes[k - 1] = num_tes[k - 1] + 1
        for j in range(y_tes.shape[0]):
            if y_tes[j] == y_pre[j]:
                num_tes_pred[y_tes[j] - 1] = num_tes_pred[y_tes[j] - 1] + 1
        Acc = num_tes_pred / (num_tes + 1e-5) * 100
        print('test_label, y_pred_test',len(test_label), len(y_pred_test))
        classification = classification_report(test_label, y_pred_test, digits=4)
        OA = 100. * accuracy_score(test_label, y_pred_test)
        Kappa = 100. * cohen_kappa_score(test_label, y_pred_test)
        toc0 = time.time()
        training_time = toc1 - tic1
        testing_time = toc2 - tic2
        runtime = toc0 - tic0
        if args.show_results:
            print(classification)
            print("OA: ", OA, " Kappa: ", Kappa)
        if args.save_results:
            end_result = {"classification":[], "kappa":[], "training_time":[], "testing_time":[]}
            end_result["classification"] = classification
            end_result["kappa"] = Kappa
            end_result["training_time"] = training_time
            end_result["testing_time"] = testing_time
            Experiment_result[0, count] = OA
            Experiment_result[1, count] = np.mean(Acc)
            Experiment_result[2, count] = Kappa
            Experiment_result[3, count] = training_time
            Experiment_result[4, count] = testing_time
            Experiment_result[5, count] = runtime
            Experiment_result[6:, count] = Acc
            data_df = pd.DataFrame(Experiment_result)
            writer = pd.ExcelWriter(path_result + 'w' + str(w) + '_' + str(int(Experiment_result[0, count]*100)) + '.xls')
            data_df.to_excel(writer, 'Acc&Time')
            writer.save()
            with open(path_result + args.results_dir + "-" + dataset_name + '.json', 'w') as fid:
                config.update(args.__dict__)
                config.update(end_result)
                json.dump(config, fid, indent=2)
        def pre(net, pre_loader):
          pre_preds = []
          net.eval()
          net.load_state_dict(torch.load(path_weight + 'model.pth'))
          with torch.no_grad():
            for data, data_LiDAR, _ in pre_loader:
              data = data.to(args.device)
              data_LiDAR = data_LiDAR.to(args.device)
              output = net(data, data_LiDAR, w)
              pre_label = torch.argmax(output, dim=1)
              pre_preds.append(pre_label.cpu().numpy().tolist())
          return pre_preds
        pre_preds = pre(net, pre_loader)
        pre_preds = [j for i in pre_preds for j in i]
        pre_preds = np.array(pre_preds).reshape((row, col))
        pre_preds = np.where(data_gt < 1, pre_preds, data_gt)
        print('pre_preds', pre_preds.shape, len(pre_preds[pre_preds>0]))
        X_result = DrawResult(pre_preds.reshape(-1).astype(int), dataset_name)
        plt.imsave(path_result + dataset_name + '_' + str(int(Experiment_result[0, count]*100)) + '.png', X_result)
    Experiment_result[:,-2]=np.mean(Experiment_result[:,0:-2],axis=1)
    Experiment_result[:,-1]=np.std(Experiment_result[:,0:-2],axis=1)
    print('Experiment_result: ', Experiment_result)
    data_df = pd.DataFrame(Experiment_result)
    writer = pd.ExcelWriter(
        path_result + dataset_name + '_w' + str(w) + '_MeanStd_' + repr(int(Experiment_result[0, -2] * 100)) + '.xls')
    data_df.to_excel(writer, 'MeanStd')
    writer.save()
