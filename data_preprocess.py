from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing

# mat_path = './imputer/Guangzhou-data-setImputed/Imputed_tensor.mat'
mat_path = './datasets/Guangzhou-data-set/tensor.mat'
#读取.mat文件
def load_mat_variable(mat_path, var_name):
    data = loadmat(mat_path)
    arr = np.squeeze(data[var_name])
    return torch.from_numpy(arr)  # 转为 PyTorch tensor

def tensor_to_station_frames(tensor):
    """
    tensor: shape = (segments, days, times)
    返回 dict: {station_id: DataFrame(columns=["time_index","speed"])}
    """
    allframes = pd.DataFrame()
    list_ = []
    for i in range(tensor.shape[0]):
        frame = pd.DataFrame()
        frame_ = []
        flat_tensor = tensor[i].reshape(-1)
        df= pd.DataFrame({
            "time_index": np.arange(len(flat_tensor)),  # 0..8781
            "speed": flat_tensor
        })
        frame_.append(df)
        frame = pd.concat(frame_)
        list_.append(frame)
    allframes = pd.concat(list_)
    return allframes

# 按时间序列对数据分组并标准化
def group_by_time(mat_path, var_name):
    """
    :param key: 按时间分组
    :param zscore: z分数标准化
    :param grouped: 将数据按时间进行分组并标准化后的结果
    :param vehicles: 把数据转换为矩阵形式
    """
    tensor = load_mat_variable(mat_path, var_name)
    frame = tensor_to_station_frames(tensor)
    values = frame.groupby('time_index')['speed'].apply(list)
    vehicles = []
    for i in range(len(values)):
        vehicles.append(values[i])
    return vehicles

vehicles = group_by_time(mat_path,"tensor")
scaler = preprocessing.MinMaxScaler()
samples = scaler.fit_transform(vehicles)

print(len(samples))
# arr = load_mat_variable(mat_path, "tensor")
# arr_RS_0 = tensor_to_station_frames(arr)
# print(arr_RS_0)


