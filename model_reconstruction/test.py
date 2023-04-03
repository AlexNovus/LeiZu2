import numpy as np
from error_evaluate import evaluate_error, evaluate_error_single
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import scipy.io as io
import openmesh as om


def visual_mesh(v, e=None, geopath=None, color_v='#1f77b4', color_e='#1f77b4', color_g='r', s=1):
    """
    mesh可视化。
    :param v: numpy.ndarray. mesh顶点。
    :param e: numpy.ndarray. mesh边。
    :param color_v: 顶点颜色。
    :param color_e: 边颜色。
    :return:
    """

    ax = plt.figure().add_subplot(projection='3d')
    im = ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=s, color=cmap(color_v))
    if e is not None:
        for pair in e:
            ax.plot(v[pair, 0], v[pair, 1], v[pair, 2], c=color_e, linewidth=0.5)
    if geopath is not None:
        for path in geopath:
            for i in range(path.shape[0] - 1):
                ax.plot(path[i: i + 2, 0], path[i: i + 2, 1], path[i: i + 2, 2], c=color_g, linewidth=2)
    ax.set(
        xlim=(-10, 10),
        ylim=(-10, 10),
        zlim=(-10, 10)
    )
    ax.grid(False)
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, label="距离误差")
    plt.show()


def visual_mesh_compare(v_list, e, color_v_list, color_e='#1f77b4', s=5):
    """
    mesh可视化。
    :param v1: numpy.ndarray. mesh顶点。
    :param v2: numpy.ndarray. mesh顶点。
    :param e: numpy.ndarray. mesh边。
    :param color_v: 顶点颜色。
    :param color_e: 边颜色。
    :return:
    """

    ax = plt.figure().add_subplot(projection='3d')
    for i, v in enumerate(v_list):
        v[:, 0] = v[:, 0] + 20 * i
        v[:, 1] = v[:, 1] + 11 * i
        for pair in e:
            ax.plot(v[pair, 0], v[pair, 1], v[pair, 2], c=color_e, linewidth=0.5)
    v = np.concatenate(v_list, axis=0)
    color_v = np.concatenate(color_v_list, axis=0)
    im = ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=s, c=color_v)
    ax.set(
        xlim=(-20, 60),
        ylim=(-5, 35),
        zlim=(-10, 10)
    )
    ax.grid(False)
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, label='距离误差', shrink=0.3)
    plt.gca().set_box_aspect((4, 2, 1))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    """画图参数设置"""
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    # plt.rcParams["axes.labelpad"] = 8

    """读取模型"""
    model = np.load('data/basic_model.npy')
    face = io.loadmat('data/testFaces.mat')['faces'] - 1
    mesh = om.TriMesh()
    mesh.add_vertices(model)
    mesh.add_faces(face)
    face = mesh.fv_indices()
    edge = mesh.ev_indices()

    """读取数据"""
    x_train = np.load('data/x_train.npy')
    x_test = np.load('data/x_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    index_test = np.load('data/index_test.npy')

    """数据预处理"""
    scaler = preprocessing.MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    """训练误差可视化"""
    # y_train_predict_lr = np.load('result/y_train_predict_lr.npy')[0]
    # y_train_predict_svr = np.load('result/y_train_predict_svr.npy')[0]
    # y_train_predict_nn = np.load('result/y_train_predict_nn.npy')[0]
    #
    #
    # cmap = plt.colormaps["viridis"]
    # # visual_mesh_compare(y_train[0].reshape(-1, 3), y_train_predict_lr[0].reshape(-1, 3), edge)
    # color_error_lr = evaluate_error_single(y_train[0], y_train_predict_lr)
    # color_error_svr = evaluate_error_single(y_train[0], y_train_predict_svr)
    # color_error_nn = evaluate_error_single(y_train[0], y_train_predict_nn)
    # visual_mesh_compare(
    #         v_list=[
    #             y_train_predict_lr.reshape(-1, 3),
    #             y_train_predict_svr.reshape(-1, 3),
    #             y_train_predict_nn.reshape(-1, 3),
    #         ],
    #         e=edge,
    #         color_v_list=[color_error_lr, color_error_svr, color_error_nn]
    # )

    """测试误差可视化"""
    # y_test_predict_lr = np.load('result/y_test_predict_lr.npy')[1]
    # y_test_predict_svr = np.load('result/y_test_predict_svr.npy')[1]
    # y_test_predict_nn = np.load('result/y_test_predict_nn.npy')[1]
    #
    # # visual_mesh_compare(y_train[0].reshape(-1, 3), y_train_predict_lr[0].reshape(-1, 3), edge)
    # color_error_lr = evaluate_error_single(y_test[1], y_test_predict_lr)
    # color_error_svr = evaluate_error_single(y_test[1], y_test_predict_svr)
    # color_error_nn = evaluate_error_single(y_test[1], y_test_predict_nn)
    # visual_mesh_compare(
    #         v_list=[
    #             y_test_predict_lr.reshape(-1, 3),
    #             y_test_predict_svr.reshape(-1, 3),
    #             y_test_predict_nn.reshape(-1, 3),
    #         ],
    #         e=edge,
    #         color_v_list=[color_error_lr, color_error_svr, color_error_nn]
    # )

    """支持向量回归"""
    y_train_predict = np.load('result/y_train_predict_nn.npy')
    error_train = evaluate_error(y_train, y_train_predict)
    print(f'error of training data: {error_train}')

    y_test_predict = np.load('result/y_test_predict_nn.npy')
    error_test = evaluate_error(y_test, y_test_predict)
    print(f'error of test data: {error_test}')
