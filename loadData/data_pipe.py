import torch
import numpy as np
import random
import yaml
from loadData import data_reader
from loadData.split_data import HyperX, sample_gt
from sklearn.preprocessing import MinMaxScaler
def set_deterministic(seed):
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def get_data(model_name="THSGR",
    path_config=None, print_config=False, print_data_info=False, patch_size = 15):
    config = yaml.load(open(path_config, "r"), Loader=yaml.FullLoader)
    dataset_name = config["data_input"]["dataset_name"]
    path_data = config["data_input"]["path_data"]
    path_data_LiDAR = config["data_input"]["path_data_LiDAR"]
    patch_size = patch_size
    split_type = config["data_split"]["split_type"]
    train_num = config["data_split"]["train_num"]
    val_num = config["data_split"]["val_num"]
    train_ratio = config["data_split"]["train_ratio"]
    val_ratio = config["data_split"]["val_ratio"]
    num_components = config["data_transforms"]["num_components"]
    batch_size = config["data_transforms"]["batch_size"]
    remove_zero_labels = config["data_transforms"]["remove_zero_labels"]
    start = config["result_output"]["data_info_start"]
    print('dataset_name: ', dataset_name)
    data, data_gt = data_reader.load_data(dataset_name, path_data=path_data, type_data=dataset_name)
    data, pca = data_reader.apply_PCA(data, num_components=num_components)
    pad_width = patch_size // 2
    img = np.pad(data, pad_width=pad_width, mode="constant", constant_values=(0))
    img = img[:, :, pad_width:img.shape[2]-pad_width]
    data_LiDAR = data_reader.load_data_LiDAR(dataset_name, path_data_LiDAR=path_data_LiDAR)
    img_LiDAR = np.pad(data_LiDAR, pad_width=pad_width, mode="constant", constant_values=(0))
    if len(img_LiDAR.shape) == 3:
        img_LiDAR = img_LiDAR[:, :, pad_width:img_LiDAR.shape[2]-pad_width]
    else:
        img_LiDAR = img_LiDAR[:, :]
    if split_type == 'number':
        gt = np.pad(data_gt, pad_width=pad_width, mode="constant", constant_values=(0))
        train_gt, test_gt = sample_gt(gt, train_num=train_num, train_ratio=train_ratio, mode=split_type)
        print('\ntrain_gt: ', train_gt.shape, train_gt.min(), train_gt.max(), train_gt[train_gt>0].shape)
        print('\ntest_gt: ', test_gt.shape, train_gt.min(), test_gt.max(), test_gt[test_gt>0].shape)
        pre_gt = np.ones((train_gt.shape[0], train_gt.shape[1]), dtype='int32')
        print('\npre_gt: ', pre_gt.shape, pre_gt.min(), pre_gt.max(), pre_gt[pre_gt>0].shape)
        train_label, test_label = [], []
        for i in range(pad_width, train_gt.shape[0]-pad_width):
            for j in range(pad_width, train_gt.shape[1]-pad_width):
                if train_gt[i][j] > 0:
                    train_label.append(train_gt[i][j])
        for i in range(pad_width, test_gt.shape[0]-pad_width):
            for j in range(pad_width, test_gt.shape[1]-pad_width):
                if test_gt[i][j] > 0:
                    test_label.append(test_gt[i][j])
        print('random number', len(test_label))
    elif split_type == 'disjoint':
        _, train_gt = data_reader.load_data(dataset_name, path_data=path_data, type_data="TRLabel")
        _, test_gt = data_reader.load_data(dataset_name, path_data=path_data, type_data="TSLabel")
        train_gt = np.pad(train_gt, pad_width=pad_width, mode="constant", constant_values=(0))
        test_gt = np.pad(test_gt, pad_width=pad_width, mode="constant", constant_values=(0))
        print('\ntrain_gt: ', train_gt.shape, train_gt.max(), train_gt[train_gt>0].shape)
        print('\ntest_gt: ', test_gt.shape, test_gt.max(), test_gt[test_gt>0].shape)
        train_label, test_label = [], []
        for i in range(pad_width, train_gt.shape[0]-pad_width):
            for j in range(pad_width, train_gt.shape[1]-pad_width):
                if train_gt[i][j] > 0:
                    train_label.append(train_gt[i][j])
        for i in range(pad_width, test_gt.shape[0]-pad_width):
            for j in range(pad_width, test_gt.shape[1]-pad_width):
                if test_gt[i][j] > 0:
                    test_label.append(test_gt[i][j])
        print(len(train_label), len(test_label))
        pre_gt = np.ones((train_gt.shape[0], train_gt.shape[1]), dtype='int32')
        print('\npre_gt: ', pre_gt.shape, pre_gt.min(), pre_gt.max(), pre_gt[pre_gt>0].shape)
    if print_config:
        print(config)
    if print_data_info:
        data_reader.data_info(train_gt, test_gt, start=start)
    train_dataset = HyperX(img, img_LiDAR, train_gt, patch_size=patch_size, flip_augmentation=False,
                            radiation_augmentation=False, mixture_augmentation=False,
                            remove_zero_labels=remove_zero_labels)
    test_dataset = HyperX(img, img_LiDAR, test_gt, patch_size=patch_size, flip_augmentation=False,
                            radiation_augmentation=False, mixture_augmentation=False,
                            remove_zero_labels=remove_zero_labels)
    pre_dataset = HyperX(img, img_LiDAR, pre_gt, patch_size=patch_size, flip_augmentation=False,
                            radiation_augmentation=False, mixture_augmentation=False,
                            remove_zero_labels=remove_zero_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False)
    pre_loader = torch.utils.data.DataLoader(
        pre_dataset,
        batch_size=batch_size,
        shuffle=False)
    return train_loader, test_loader, train_label, test_label, pre_loader, data_gt, train_dataset
