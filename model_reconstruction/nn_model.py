import numpy as np
from sklearn import preprocessing
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt


class FNN(nn.Module):

    def __init__(self, layer_sizes):
        super(FNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.layers.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            if i != len(layer_sizes) - 1:
                self.layers.append(nn.BatchNorm1d(layer_sizes[i]))
                self.layers.append(torch.nn.ELU(True))
        # self.active = torch.nn.ELU(True)
        #
        # self.linear1 = torch.nn.Linear(24, 4096)
        # self.batch1 = torch.nn.BatchNorm1d(4096)
        #
        # self.linear2 = torch.nn.Linear(4096, 8192)
        # self.batch2 = torch.nn.BatchNorm1d(8192)
        #
        # self.linear3 = torch.nn.Linear(8192, 4096)
        # self.batch3 = torch.nn.BatchNorm1d(4096)
        #
        # self.linear4 = torch.nn.Linear(4096, 8192)
        # self.batch4 = torch.nn.BatchNorm1d(8192)
        #
        # self.linear5 = torch.nn.Linear(8192, 4096)
        # self.batch5 = torch.nn.BatchNorm1d(4096)
        #
        # self.linear6 = torch.nn.Linear(4096, 4041)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            print(x.shape)
        # x = self.active(self.batch1(self.linear1(x)))
        # x = self.active(self.batch3(self.linear3(self.active(self.batch2(self.linear2(x))))) + x)
        # x = self.active(self.batch5(self.linear5(self.active(self.batch4(self.linear4(x))))) + x)
        # x = self.linear6(x)

        return x


class Data(Dataset):

    def __init__(self, x, y, device):
        self.x = torch.from_numpy(x).float().to(device)
        self.y = torch.from_numpy(y).float().to(device)
        x_len = x.shape[0]
        y_len = y.shape[0]
        if x_len != y_len:
            print('x and y have different sizes for their first dimension')
        else:
            self.len = x_len

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # elif isinstance(m. nn.BatchNorm1d):
    #     nn.init.constant_(m.weight, 1)
    #     nn.init.constant_(m.bias, 0)


def show_error(ground_truth, prediction, index_gt):
    """根据真实值和预测值计算误差"""

    error = ground_truth - prediction
    num_samples = np.shape(error)[0]
    num_vertices = int(np.shape(error)[1] / 3)

    average_distance_error_all = np.zeros(num_samples) # 所有模型的平均距离误差
    max_distance_error_all = np.zeros(num_samples)  # 所有模型的最大距离误差
    average_relative_distance_error_all = np.zeros(num_samples)  # 所有模型的平均相对误差
    max_relative_distance_error_all = np.zeros(num_samples)  # 所有模型的最大相对误差

    for index_model in range(num_samples):

        y = ground_truth[index_model].reshape((num_vertices, 3))[:, 1]
        length = np.max(y) - np.min(y)

        distance_error_one = np.zeros(num_vertices)  # 该模型所有顶点距离误差

        for index_vertex in range(num_vertices):
            l1 = error[index_model, 3 * index_vertex]
            l2 = error[index_model, 3 * index_vertex + 1]
            l3 = error[index_model, 3 * index_vertex + 2]
            distance_error_one[index_vertex] = (l1 ** 2 + l2 ** 2 + l3 ** 2) ** 0.5

        mean_one = np.mean(distance_error_one)
        max_one = np.max(distance_error_one)
        average_distance_error_all[index_model] = mean_one  # 该模型平均距离误差
        max_distance_error_all[index_model] = max_one  # 该模型最大距离误差
        average_relative_distance_error_all[index_model] = mean_one / length * 100  # 该模型平均相对误差
        max_relative_distance_error_all[index_model] = max_one / length * 100  # 该模型最大相对误差

    print('average distance error:', np.mean(average_distance_error_all))
    print(index_gt[np.argmax(max_distance_error_all)], 'has max distance error:', np.max(max_distance_error_all))
    print('average relative distance error:', np.mean(average_relative_distance_error_all), '%')
    print(index_gt[np.argmax(max_relative_distance_error_all)], 'has max relative distance error:',
          np.max(max_relative_distance_error_all), '%')

    return


def calculate_error(ground_truth, prediction):
    model_diff = ground_truth - prediction
    error = []
    for diff in model_diff:
        a = list(map(np.linalg.norm, diff.reshape(-1, 3)))
        error.append(np.mean(a))
    a = np.mean(error)
    print(a)
    return a


"""划分训练集验证集测试集"""
x_train = np.load('data/x_train.npy')
x_test = np.load('data/x_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')
index_test = np.load('data/index_test.npy')

"""数据预处理"""
scaler = preprocessing.MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


"""超参数 24, 512, 4096, 4041"""
layer_sizes = [24, 64, 256, 1024, 4096, 8192, 4041]
batch_size = 32
learning_rate = 0.0000001
epoch = 0
penalty = 0.00001

"""训练网络"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

net = FNN(layer_sizes).to(device)
net.apply(init_weight)
# net.load_state_dict(torch.load('nn_parameter.pkl'))
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=penalty)

train_data = Data(x=x_train, y=y_train, device=device)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

min_error = 100
model_parameter = None
for i in range(epoch):
    for j, (x, y_) in enumerate(train_loader):
        y = net(x)
        loss_func = torch.nn.MSELoss()
        loss = loss_func(y, y_)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # if i % 5 == 0 or i == epoch - 1:
    #     print(i)
    #     print(loss.item())
    #     with torch.no_grad():
    #         y_train_predict = net(torch.from_numpy(x_train).float().to(device)).detach().cpu().numpy()
    #         y_test_predict = net(torch.from_numpy(x_test).float().to(device)).detach().cpu().numpy()
    #     calculate_error(y_train, y_train_predict)
    #     test_error = calculate_error(y_test, y_test_predict)
    #     print('\n')
    #     if test_error < min_error:
    #         min_error = test_error
    #         model_parameter = net.state_dict()
    #
    # if i % 500 == 0 or i == epoch - 1:
    #     print('min error:', min_error)
    #     print('\n')
    #     torch.save(model_parameter, 'nn_parameter.pkl')

# net.load_state_dict(torch.load('data/nn_parameter.pkl'))
with torch.no_grad():
    y_train_predict_nn = net(torch.from_numpy(x_train).float().to(device)).detach().cpu().numpy()
    y_test_predict_nn = net(torch.from_numpy(x_test).float().to(device)).detach().cpu().numpy()
calculate_error(y_train, y_train_predict_nn)
calculate_error(y_test, y_test_predict_nn)
np.save('y_train_predict_nn.npy', y_train_predict_nn)
np.save('y_test_predict_nn.npy', y_test_predict_nn)
