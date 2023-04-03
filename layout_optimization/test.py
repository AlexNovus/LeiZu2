from layout import *
from cvt import *


def visual(data, label_x, label_y):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=50)
    ax.hist(data, 16)
    ax.set(
        xlabel=label_x,
        ylabel=label_y,
        ylim=(0, 20),
        yticks=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    )
    plt.show()


def visual_res(v, f, e, fg, cg, pair):

    v_generator = []
    for i in range(fg.shape[0]):
        v_generator.append(np.dot(cg[i], v[fg[i]]))
    v_generator = np.array(v_generator)
    v, f = insert_vertex(v, f, v_generator)
    v[:, 1] = -v[:, 1]
    d, p = geodesic_path(v, f, pair)
    print(max(d))
    print(min(d))
    print(np.mean(d))
    print(max(map(lambda x: abs(x - 7.5) / 7.5 * 100, d)))
    visual_mesh(v=v, e=e, geopath=p)



if __name__ == '__main__':

    """画图参数设置"""
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 20
    plt.rcParams["axes.labelpad"] = 8

    # 读取面信息
    face = io.loadmat('testFaces.mat')['faces'] - 1
    num_vertices = np.max(face) + 1

    # 读取坐标信息
    model = np.load('basic_model.npy')

    # 生成obj
    mesh = om.TriMesh()
    mesh.add_vertices(model)
    mesh.add_faces(face)
    face = mesh.fv_indices()
    edge = mesh.ev_indices()

    """单线优化结果可视化"""
    res = np.load('single_res.npy')
    print(len(np.where((res >= 9.99) & (res <= 10.01))[0]))
    # visual(res, '长度', '个数')

    """六芒星网络优化结果可视化"""
    layout_size = 75
    f_generator = np.load(f'result/layout_hexagram_{layout_size}_face.npy')
    c_generator = np.load(f'result/layout_hexagram_{layout_size}_coefficient.npy')
    feature_point_pair = np.array([[0, 1],
                                   [0, 2],
                                   [0, 3],
                                   [0, 4],
                                   [0, 5],
                                   [0, 6],
                                   [1, 2],
                                   [2, 3],
                                   [3, 4],
                                   [4, 5],
                                   [5, 6],
                                   [1, 6],
                                   [1, 7],
                                   [1, 8],
                                   [2, 8],
                                   [2, 9],
                                   [3, 9],
                                   [3, 10],
                                   [4, 10],
                                   [4, 11],
                                   [5, 11],
                                   [5, 12],
                                   [6, 12],
                                   [6, 7]]) + 1347
    visual_res(model, face, edge, f_generator, c_generator, feature_point_pair)
