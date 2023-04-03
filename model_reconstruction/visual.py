import numpy as np
import scipy.io as io
import openmesh as om
from sklearn import preprocessing
import igl
import pygeodesic.geodesic as geodesic
import matplotlib.pyplot as plt
import joblib


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
    ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=s, c=color_v)
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
    # plt.savefig('test.svg', bbox_inches='tight')
    plt.show()


def calculate_error(ground_truth, prediction):
    model_diff = ground_truth - prediction
    error = []
    for diff in model_diff:
        a = list(map(np.linalg.norm, diff.reshape(-1, 3)))
        error.append(np.mean(a))
    print(np.mean(error))


"""读取数据"""
x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

"""数据预处理"""
scaler = preprocessing.MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

lr = joblib.load('linear_model.pkl')
model = lr.predict(x_train[0:1]).reshape(-1, 3)
# calculate_error(y_train, model)

face = io.loadmat('testFaces.mat')['faces'] - 1
mesh = om.TriMesh()
mesh.add_vertices(model)
mesh.add_faces(face)
face = mesh.fv_indices()
edge = mesh.ev_indices()
visual_mesh(model, edge)
