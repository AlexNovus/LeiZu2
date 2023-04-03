import numpy as np
import scipy.io as io
import openmesh as om
import igl
import pygeodesic.geodesic as geodesic
import matplotlib.pyplot as plt


def visual_mesh(v, e=None, geopath=None, color_v='#1f77b4', color_e='#1f77b4', color_g='r', s=1):
    """
    mesh可视化。
    :param v: numpy.ndarray. mesh顶点。
    :param e: numpy.ndarray. mesh边。
    :param color_v: 顶点颜色。
    :param color_e: 边颜色。
    :return:
    """

    ax = plt.figure(figsize=(8, 8), dpi=50).add_subplot(projection='3d')
    ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=s, c=color_v)
    if e is not None:
        for pair in e:
            ax.plot(v[pair, 0], v[pair, 1], v[pair, 2], c=color_e, linewidth=0.5)
    if geopath is not None:
        for path in geopath:
            for i in range(path.shape[0] - 1):
                ax.plot(path[i: i + 2, 0], path[i: i + 2, 1], path[i: i + 2, 2], c=color_g, linewidth=3)
    ax.set(
        xlim=(-10, 10),
        ylim=(-10, 10),
        zlim=(-10, 10)
    )
    ax.grid(False)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def insert_vertex(v, f, v_insert):
    """
    mesh插入顶点。
    :param v: numpy.ndarray. mesh顶点。
    :param f: numpy.ndarray. mesh面。
    :param v_insert: numpy.ndarray. 插入顶点。
    :return: numpy.ndarray. numpy.ndarray. mesh顶点；mesh面。
    """

    """基本定义"""
    num_v = v.shape[0]  # 原顶点个数，也即新顶点编号
    v = np.concatenate((v, v_insert), axis=0)  # 插入新顶点

    """修改面"""
    for vi in v_insert:

        """判断插入面编号"""
        # dist = np.zeros(num_v)
        # for j in range(num_v):
        #     dist[j] = np.linalg.norm(v_insert[i] - v[j])
        # v_index = np.argsort(dist)
        # f_index = list(map(set, f)).index(set(v_index[:3]))
        _, f_index, _ = igl.point_mesh_squared_distance(vi, v, f)
        v1, v2, v3 = f[f_index]
        # visual_mesh(v[[v1, v2, v3, num_v]], e=[[0, 1], [1, 2], [2, 0]])
        f = np.delete(f, f_index, axis=0)  # 删除旧面
        f = np.concatenate((
            f, np.array([
                [v1, v2, num_v],
                [v2, v3, num_v],
                [v3, v1, num_v]
            ])
        ), axis=0)  # 插入新面
        num_v += 1  # 顶点个数加1

    return v, f


def voronoi_delaunay_triangulation(v, f, e, v_generator):

    num_v = v.shape[0]
    num_g = v_generator.shape[0]

    v, f = insert_vertex(v=v, f=f, v_insert=v_generator)

    m = om.TriMesh()
    m.add_vertices(v)
    m.add_faces(f)
    f = m.fv_indices()

    geodist = np.zeros((num_v, num_g))

    for i in range(num_v):
        geodist[i, :] = igl.exact_geodesic(
            v=v, f=f, vs=np.array([i]), vt=np.arange(num_v, num_v + num_g), fs=None, ft=None
        )

    """划分voronoi"""
    voronoi = np.concatenate((np.argmin(geodist, axis=1), np.arange(num_g)), axis=0)

    """生成连接"""
    connection = []
    for v1, v2 in e:
        if voronoi[v1] != voronoi[v2]:
            pair = sorted([num_v + voronoi[v1], num_v + voronoi[v2]])
            if pair not in connection:
                connection.append(pair)
    connection = np.array(connection)

    return v, f, voronoi, connection


def geodesic_path(v, f, v_pair):
    geoalg = geodesic.PyGeodesicAlgorithmExact(v, f)
    geodist = []
    geopath = []
    for v1, v2 in v_pair:
        dist, path = geoalg.geodesicDistance(v1, v2)
        geodist.append(dist)
        geopath.append(path)

    return sorted(geodist), geopath


if __name__ == '__main__':

    """读取模型"""
    model = np.load('basic_model.npy')
    face = io.loadmat('testFaces.mat')['faces'] - 1
    mesh = om.TriMesh()
    mesh.add_vertices(model)
    mesh.add_faces(face)
    face = mesh.fv_indices()
    edge = mesh.ev_indices()

    """模型可视化"""
    # visual_mesh(v=model, e=edge, color_v='#1f77b4', color_e='#1f77b4')

    centroid_5 = np.array([
        [1.27486, -8.68652, 4.31078],
        [3.58475, 2.29998, -0.843612],
        [-2.60809, -0.898292, 2.51371],
        [-1.27729, 8.38807, -3.1983],
        [0.284716, -6.70536, -3.07034],
    ])
    centroid_6 = np.array([
        [-1.7249, 9.44342, -1.43069],
        [0.904506, -3.62799, -2.9673],
        [-2.1913, 1.9897, 1.31466],
        [2.63631, 4.67678, -3.91535],
        [1.3276, -9.03013, 2.28855],
        [-3.20082, -4.14643, 5.15428],
    ])
    centroid_7 = np.array([
        [-0.481959, -7.13803, -2.94902],
        [-3.77429, 2.35082, -0.35126],
        [-2.01727, 9.61333, -1.48196],
        [2.15081, -0.948195, -3.25306],
        [-0.784409, -1.57543, 4.72788],
        [3.00115, 6.21322, -1.38],
        [1.05729, -9.04458, 3.84219],
    ])
    centroid_8 = np.array([
        [0.457285, -1.21294, -2.96813],
        [-2.28567, -8.15728, 1.62828],
        [1.51301, -8.40492, 5.1827],
        [-1.02764, 9.68167, -3.02032],
        [-3.8244, 3.66534, -0.504535],
        [1.57116, -7.93408, -3.03192],
        [-0.678936, -1.32685, 4.36974],
        [3.05395, 4.83197, -1.00365],
    ])
    centroid_9 = np.array([
        [0.530223, -7.59183, -3.12421],
        [-2.34099, 3.87422, -3.06429],
        [-2.97181, 9.86706, -1.41833],
        [2.99895, 5.75038, -1.24702],
        [-1.64706, -9.55939, 3.645],
        [-3.54217, -1.41404, 0.696152],
        [1.50274, -3.09669, 5.23006],
        [0.387415, 9.99995, -3.30956],
        [3.57364, -0.608732, -0.814577],
    ])
    centroid_10 = np.array([
        [1.30106, 7.99458, -3.6322],
        [-3.42835, 1.22432, 0.459119],
        [-3.26489, 6.53135, -0.210597],
        [2.46589, -8.72443, -0.369461],
        [1.03267, -2.67069, 5.51026],
        [-1.61334, 10.7673, -2.95857],
        [3.72966, -1.36454, -1.16153],
        [-1.98109, -9.07271, 4.56787],
        [-2.39621, -5.32648, -2.00518],
        [3.39201, 4.03225, -0.972899],
    ])
    centroid_11 = np.array([
        [0.826894, -8.68787, -3.03893],
        [2.68126, -3.69404, 1.70765],
        [3.39336, 3.65654, -0.889304],
        [-3.3506, 2.8095, 0.279815],
        [1.6203, 6.94305, -3.75188],
        [-2.25445, -2.46828, 5.69698],
        [0.399216, 11.0572, -2.04447],
        [-3.31003, -4.71556, 0.135407],
        [1.1488, -0.733538, -3.11177],
        [0.765785, -9.50612, 4.79202],
        [-2.84237, 9.21239, -1.18883],
    ])
    centroid_12 = np.array([
        [-2.13774, 6.27923, 0.00934143],
        [1.66692, -8.24385, 5.63965],
        [-0.00344402, 10.3093, -3.23075],
        [2.56156, 0.718638, -3.48536],
        [-3.8154, 1.04315, -1.03809],
        [1.4426, 0.871742, 1.78167],
        [-1.56773, -5.06537, -2.46407],
        [-3.4371, 10.5051, -1.54733],
        [-2.96526, -3.23332, 4.95853],
        [2.98705, -5.91655, -0.248694],
        [2.66755, 6.31961, -3.80761],
        [-1.57661, -10.0189, 0.395026],
    ])
    centroid_13 = np.array([
        [-2.97181, 9.86706, -1.41833],
        [-3.10725, -4.24751, 5.75048],
        [2.6915, 6.70992, -1.41058],
        [-3.15282, -2.75941, -1.29949],
        [2.09985, -1.82845, -3.15807],
        [1.31856, -8.67583, 5.28877],
        [1.15298, 10.2351, -2.06135],
        [2.89081, -7.25413, -0.513865],
        [3.92821, 2.83491, -1.19487],
        [1.07388, -0.997678, 3.14607],
        [-3.23315, 3.13375, 0.352502],
        [-1.44118, 5.3376, -3.36693],
        [-1.84169, -9.64935, -0.213746],
    ])
    centroid_14 = np.array([
        [3.35618, -3.46026, -0.597312],
        [3.17124, 7.0066, -1.65289],
        [-3.35689, 3.54185, 0.157218],
        [0.615898, 10.5632, -1.99369],
        [2.36863, -9.49074, -0.9085],
        [3.58475, 2.29998, -0.843612],
        [-1.46782, -1.93765, -2.5557],
        [-2.63143, 7.61392, -0.54234],
        [0.506408, -2.3798, 5.83189],
        [-0.448637, 4.35024, -3.38137],
        [-2.68532, -7.10051, 0.397269],
        [1.02893, -9.07496, 4.86818],
        [-1.93937, -0.0907674, 2.60485],
        [-3.0158, 10.4177, -3.18005],
    ])
    centroid_15 = np.array([
        [-2.55034, 11.2983, -3.06238],
        [1.15298, 10.2351, -2.06135],
        [3.31734, -4.68264, -0.886821],
        [-0.459951, 5.23655, -3.46478],
        [-3.05053, -3.1332, 4.34095],
        [-3.23315, 3.13375, 0.352502],
        [2.55774, -5.44674, 5.4832],
        [-2.9402, 7.81709, -0.612501],
        [-1.92731, -1.42926, -2.46502],
        [-2.02587, -9.03337, 5.3075],
        [3.90068, 1.26464, -3.25714],
        [-2.1257, -7.37377, -1.93429],
        [3.32838, 6.19937, -1.50066],
        [1.49948, -10.2384, 0.369426],
        [1.15231, 0.280369, 2.34274],
    ])

    # visual_mesh(v=model, e=edge, color_v='#1f77b4', color_e='#1f77b4')
    # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    centroid = centroid_15
    v, f, voro, conn = voronoi_delaunay_triangulation(model, face, edge, centroid)
    d, p = geodesic_path(v, f, conn)
    # print(round(max(d), 2), round(min(d), 2), round(np.mean(d), 2))
    # visual_mesh(v=v, e=edge, color_v=voro, s=20)
    visual_mesh(v=v, e=edge, geopath=p)
