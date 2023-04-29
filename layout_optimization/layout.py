import numpy as np
import openmesh as om
import scipy.io as io
import igl
import matplotlib.pyplot as plt
import pygeodesic.geodesic as geodesic


def determine_face(origin_faces, previous_face, coefficient):
    """

    :param origin_faces:
    :param previous_face:
    :param coefficient:
    :return:
    """
    """确定新系数下点所在面"""

    # 系数为负则在该顶点对面
    now_face = previous_face
    for index in range(3):
        if coefficient[index] < 0:
            # 系数非负的顶点
            index0 = (1 + index) % 3
            index1 = (2 + index) % 3
            # 系数为负的顶点
            index2 = index % 3
            # 系数非负顶点相邻的面
            f0 = set(np.where(origin_faces == previous_face[index0])[0])
            f1 = set(np.where(origin_faces == previous_face[index1])[0])
            # 系数为负顶点相邻的面
            f2 = set(np.where(origin_faces == previous_face[index2])[0])
            # 系数为负顶点没有在对面的面
            if len(list(f0 & f1)) == 1:
                break
            # 存在对面的面
            fi = list((f0 & f1) - f2)[0]
            now_face = origin_faces[fi]
            break

    return now_face


def add_delete_faces(origin_faces, change_face, add_vertex):
    """当新添加点在面上时，添加并删除面"""

    # 所改变的面的序号
    f0 = set(np.where(origin_faces == change_face[0])[0])
    f1 = set(np.where(origin_faces == change_face[1])[0])
    f2 = set(np.where(origin_faces == change_face[2])[0])
    fi = list(f0 & f1 & f2)[0]

    # 删除该面
    new_faces = np.delete(origin_faces, fi, axis=0)

    # 添加面
    add_faces = np.array([[change_face[0], change_face[1], add_vertex],
                          [change_face[1], change_face[2], add_vertex],
                          [change_face[2], change_face[0], add_vertex]])
    new_faces = np.concatenate((new_faces, add_faces), axis=0)

    return new_faces


def geodesic_distance(
        vertices,
        faces,
        face_1,
        coefficient_1,
        face_2,
        coefficient_2):
    """在原网格上添加新的点和面，求测地距离"""

    # 添加新的点
    position_1 = coefficient_1.dot(vertices[face_1])
    position_2 = coefficient_2.dot(vertices[face_2])
    index_1 = np.shape(vertices)[0]
    index_2 = index_1 + 1
    new_vertices = np.concatenate(
        (vertices, np.array(
            [position_1]), np.array(
            [position_2])), axis=0)

    # 修改面
    new_faces = add_delete_faces(
        origin_faces=faces,
        change_face=face_1,
        add_vertex=index_1)
    new_faces = add_delete_faces(
        origin_faces=new_faces,
        change_face=face_2,
        add_vertex=index_2)

    # 计算测地距离
    mesh = om.TriMesh()
    mesh.add_vertices(new_vertices)
    mesh.add_faces(new_faces)
    new_faces = mesh.fv_indices()
    geodesic_distance = igl.exact_geodesic(
        v=new_vertices, f=new_faces, vs=np.array(
            [index_1]), vt=np.array(
            [index_2]))

    return np.array([geodesic_distance])[0]


def numerical_gradient(
        vertices,
        faces,
        face_1,
        coefficient_1,
        face_2,
        coefficient_2):
    """损失函数对参数化坐标求数值微分"""

    # 数值梯度步长
    delta = 0.01

    # 原始测地距离
    gd_0 = geodesic_distance(
        vertices=vertices,
        faces=faces,
        face_1=face_1,
        coefficient_1=coefficient_1,
        face_2=face_2,
        coefficient_2=coefficient_2)

    # 点1，a方向
    coefficient_1_a = coefficient_1.copy()
    if coefficient_1_a[0] + delta > 1:
        coefficient_1_a[0] = coefficient_1_a[0] + coefficient_1_a[2]
    else:
        coefficient_1_a[0] = coefficient_1_a[0] + delta
    delta_s = coefficient_1_a[2]
    coefficient_1_a[2] = 1 - coefficient_1_a[0] - coefficient_1_a[1]
    gd_1_a = geodesic_distance(
        vertices=vertices,
        faces=faces,
        face_1=face_1,
        coefficient_1=coefficient_1_a,
        face_2=face_2,
        coefficient_2=coefficient_2)
    difference_1_a = (gd_1_a - gd_0) / min(delta, delta_s)

    # 点1，b方向
    coefficient_1_b = coefficient_1.copy()
    if coefficient_1_b[1] + delta > 1:
        coefficient_1_b[1] = coefficient_1_b[1] + coefficient_1_b[2]
    else:
        coefficient_1_b[1] = coefficient_1_b[1] + delta
    delta_s = coefficient_1_b[2]
    coefficient_1_b[2] = 1 - coefficient_1_b[0] - coefficient_1_b[1]
    gd_1_b = geodesic_distance(
        vertices=vertices,
        faces=faces,
        face_1=face_1,
        coefficient_1=coefficient_1_b,
        face_2=face_2,
        coefficient_2=coefficient_2)
    difference_1_b = (gd_1_b - gd_0) / min(delta, delta_s)

    # 点2，a方向
    coefficient_2_a = coefficient_2.copy()
    if coefficient_2_a[0] + delta > 1:
        coefficient_2_a[0] = coefficient_2_a[0] + coefficient_2_a[2]
    else:
        coefficient_2_a[0] = coefficient_2_a[0] + delta
    delta_s = coefficient_2_a[2]
    coefficient_2_a[2] = 1 - coefficient_2_a[0] - coefficient_2_a[1]
    gd_2_a = geodesic_distance(
        vertices=vertices,
        faces=faces,
        face_1=face_1,
        coefficient_1=coefficient_1,
        face_2=face_2,
        coefficient_2=coefficient_2_a)
    difference_2_a = (gd_2_a - gd_0) / min(delta, delta_s)

    # 点2，b方向
    coefficient_2_b = coefficient_2.copy()
    if coefficient_2_b[1] + delta > 1:
        coefficient_2_b[1] = coefficient_2_b[1] + coefficient_2_b[2]
    else:
        coefficient_2_b[1] = coefficient_2_b[1] + delta
    delta_s = coefficient_2_b[2]
    coefficient_2_b[2] = 1 - coefficient_2_b[0] - coefficient_2_b[1]
    gd_2_b = geodesic_distance(
        vertices=vertices,
        faces=faces,
        face_1=face_1,
        coefficient_1=coefficient_1,
        face_2=face_2,
        coefficient_2=coefficient_2_b)
    difference_2_b = (gd_2_b - gd_0) / min(delta, delta_s)

    return gd_0, difference_1_a, difference_1_b, difference_2_a, difference_2_b


def gradient_descent_method(
        vertices,
        faces,
        faces_generator,
        coefficients_generator,
        generator_pairs,
        setting_length,
        step_length):
    """梯度下降法微调优化"""

    # 生成元个数
    num_generators = np.shape(faces_generator)[0]

    # 梯度初始化
    gradient = np.zeros((num_generators, 3))

    # 计算梯度
    for pair in generator_pairs:
        index_1 = pair[0]
        index_2 = pair[1]
        gd, d1a, d1b, d2a, d2b = numerical_gradient(vertices=vertices, faces=faces,
                                                    face_1=faces_generator[index_1],
                                                    coefficient_1=coefficients_generator[index_1],
                                                    face_2=faces_generator[index_2],
                                                    coefficient_2=coefficients_generator[index_2])
        gradient[index_1, 0] += 2 * (gd - setting_length) * d1a
        gradient[index_1, 1] += 2 * (gd - setting_length) * d1b
        gradient[index_2, 0] += 2 * (gd - setting_length) * d2a
        gradient[index_2, 1] += 2 * (gd - setting_length) * d2b

    # 更新参数
    l = np.sum(gradient * gradient) ** 0.5
    # print('length of gradient:')
    # print(' ', l)
    new_coefficients_generator = coefficients_generator - step_length * gradient / l
    new_faces_generator = faces_generator.copy()
    for index in range(num_generators):
        new_coefficients_generator[index,
                                   2] = 1 - new_coefficients_generator[index,
                                                                       0] - new_coefficients_generator[index,
                                                                                                       1]
        new_faces_generator[index] = determine_face(
            origin_faces=faces,
            previous_face=faces_generator[index],
            coefficient=new_coefficients_generator[index])
        if (new_coefficients_generator[index] < 0).any():
            new_coefficients_generator[index] = np.array([1 / 3, 1 / 3, 1 / 3])

    # 计算新测地距离和长度损失
    loss = 0
    max_relative_error = 0
    # print('geodesic distances:')
    all_gd = []
    for pair in generator_pairs:
        index_1 = pair[0]
        index_2 = pair[1]
        gd = geodesic_distance(
            vertices=vertices,
            faces=faces,
            face_1=new_faces_generator[index_1],
            coefficient_1=new_coefficients_generator[index_1],
            face_2=new_faces_generator[index_2],
            coefficient_2=new_coefficients_generator[index_2])
        # print(' ', gd)
        all_gd.append(gd)
        loss += (gd - setting_length) ** 2
        absolute_error = abs(gd - setting_length)
        if absolute_error > max_relative_error:
            max_relative_error = absolute_error
    max_relative_error = max_relative_error / setting_length * 100
    # print('loss:')
    # print(' ', loss)

    return new_faces_generator, new_coefficients_generator, max_relative_error, all_gd


def single_test(v, f, initial_f):
    # 单根线测试
    # 定义初始位置和连接
    faces_generator = f[initial_f]
    coefficients_generator = np.ones((2, 3)) / 3
    feature_point_pair = np.array([[0, 1]])
    # 初始测地长度
    gd = geodesic_distance(
        vertices=v,
        faces=f,
        face_1=faces_generator[0],
        coefficient_1=coefficients_generator[0],
        face_2=faces_generator[1],
        coefficient_2=coefficients_generator[1])
    # print(f'geodesic distances: {gd}')
    # 开始迭代计算
    target_length = 10
    iteration = 0
    for i in range(100):
        if i < 20:
            step_length = 0.5
        elif i < 40:
            step_length = 0.2
        elif i < 60:
            step_length = 0.1
        elif i < 80:
            step_length = 0.05
        else:
            step_length = 0.02
        # print(f'iterations: {iteration}')
        faces_generator, coefficients_generator, error, all_gd = \
            gradient_descent_method(vertices=v, faces=f, faces_generator=faces_generator,
                                    coefficients_generator=coefficients_generator, generator_pairs=feature_point_pair,
                                    setting_length=target_length, step_length=step_length)
        iteration += 1
        # print('max relative error:')
        # print(' ', error, '\n')

    return all_gd[0]

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

    # # 单根线测试
    # # 定义初始位置和连接
    # faces_generator = face[[0, 11]]
    # coefficients_generator = np.ones((2, 3)) / 3
    # feature_point_pair = np.array([[0, 1]])
    # # 初始测地长度
    # gd = geodesic_distance(
    #     vertices=model,
    #     faces=face,
    #     face_1=faces_generator[0],
    #     coefficient_1=coefficients_generator[0],
    #     face_2=faces_generator[1],
    #     coefficient_2=coefficients_generator[1])
    # print(f'geodesic distances: {gd}')
    # # 开始迭代计算
    # target_length = 10
    # iteration = 0
    # for i in range(100):
    #     if i < 20:
    #         step_length = 0.5
    #     elif i < 40:
    #         step_length = 0.2
    #     elif i < 60:
    #         step_length = 0.1
    #     elif i < 80:
    #         step_length = 0.05
    #     else:
    #         step_length = 0.02
    #     print(f'iterations: {iteration}')
    #     faces_generator, coefficients_generator, error = \
    #         gradient_descent_method(vertices=model, faces=face, faces_generator=faces_generator,
    #                                 coefficients_generator=coefficients_generator, generator_pairs=feature_point_pair,
    #                                 setting_length=target_length, step_length=step_length)
    #     iteration += 1
    #     print('max relative error:')
    #     print(' ', error, '\n')
    # single_test(v=model, f=face, initial_f=[0, 11])
    # res = []
    # for i in range(100):
    #     print(i, end=': ')
    #     np.random.seed(i)
    #     res.append(single_test(v=model, f=face, initial_f=np.random.randint(0, face.shape[0], 2).tolist()))
    #     print(res[-1])
    # res = np.array(res)
    # np.save('single_res.npy', res)

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


    def geodesic_path(v, f, v_pair):
        geoalg = geodesic.PyGeodesicAlgorithmExact(v, f)
        geodist = []
        geopath = []
        for v1, v2 in v_pair:
            dist, path = geoalg.geodesicDistance(v1, v2)
            geodist.append(dist)
            geopath.append(path)

        return sorted(geodist), geopath


    def visual_res(v, f, e, fg, cg, pair):

        v_generator = []
        for i in range(13):
            v_generator.append(np.dot(cg[i], v[fg[i]]))
        v_generator = np.array(v_generator)
        print(v_generator)
        v, f = insert_vertex(v, f, v_generator)
        d, p = geodesic_path(v, f, pair)
        visual_mesh(v=v, e=e, geopath=p)

    # 复杂网络测试
    # 定义初始位置和连接
    # faces_generator = face[[1955, 793, 1256, 2016, 2061, 1902, 772, 386, 551, 1293, 2198, 1712, 829]]
    # coefficients_generator = np.ones((13, 3)) / 3

    # feature_point_pair = np.array([[0, 1],
    #                                [0, 2],
    #                                [0, 3],
    #                                [0, 4],
    #                                [0, 5],
    #                                [0, 6],
    #                                [1, 2],
    #                                [2, 3],
    #                                [3, 4],
    #                                [4, 5],
    #                                [5, 6],
    #                                [1, 6],
    #                                [1, 7],
    #                                [1, 8],
    #                                [2, 8],
    #                                [2, 9],
    #                                [3, 9],
    #                                [3, 10],
    #                                [4, 10],
    #                                [4, 11],
    #                                [5, 11],
    #                                [5, 12],
    #                                [6, 12],
    #                                [6, 7]]) + num_vertices
    # visual_res(model, face, edge, faces_generator, coefficients_generator, feature_point_pair)

    # feature_point_pair = np.array([[0, 1],
    #                                [0, 2],
    #                                [0, 3],
    #                                [0, 4],
    #                                [0, 5],
    #                                [0, 6],
    #                                [1, 2],
    #                                [2, 3],
    #                                [3, 4],
    #                                [4, 5],
    #                                [5, 6],
    #                                [1, 6],
    #                                [1, 7],
    #                                [1, 8],
    #                                [2, 8],
    #                                [2, 9],
    #                                [3, 9],
    #                                [3, 10],
    #                                [4, 10],
    #                                [4, 11],
    #                                [5, 11],
    #                                [5, 12],
    #                                [6, 12],
    #                                [6, 7]])
    #
    # # 开始迭代计算
    # target_length = 7.5
    # iteration = 0
    # min_error = 100
    # min_face = faces_generator
    # min_coefficient = coefficients_generator
    # for i in range(1000):
    #     if i < 200:
    #         step_length = 0.5
    #     elif i < 400:
    #         step_length = 0.2
    #     elif i < 600:
    #         step_length = 0.1
    #     elif i < 800:
    #         step_length = 0.05
    #     else:
    #         step_length = 0.02
    #     print(f'iterations: {iteration}')
    #     faces_generator, coefficients_generator, error, all_gd = \
    #         gradient_descent_method(vertices=model, faces=face, faces_generator=faces_generator,
    #                                 coefficients_generator=coefficients_generator,
    #                                 generator_pairs=feature_point_pair,
    #                                 setting_length=target_length, step_length=step_length)
    #     iteration += 1
    #     print('max relative error:')
    #     print(' ', error, '\n')
    #     if error < min_error:
    #         min_error = error
    #         min_face = faces_generator
    #         min_coefficient = coefficients_generator
    #
    # np.save(f'result/layout_hexagram_{int(target_length*10)}_face.npy', min_face)
    # np.save(f'result/layout_hexagram_{int(target_length*10)}_coefficient.npy', min_coefficient)

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
                                   [6, 7]]) + num_vertices
    visual_res(model, face, edge, faces_generator, coefficients_generator, feature_point_pair)
