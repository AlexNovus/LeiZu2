import numpy as np
import torch
from scipy.optimize import minimize


"""n*n网络，正问题"""


def conductance_to_kirchhoff_to_response(conductance_row, conductance_column):
    """
    根据给定的电导，计算基尔霍夫矩阵和响应矩阵。电阻网络的规模（n*n）由给定的电导（n*(n+1)和(n+1)*n）决定。
    :param conductance_row: numpy.ndarray. 行电导，大小为n*(n+1)。
    :param conductance_column: numpy.ndarray. 列电导，大小为(n+1)*n。
    :return: numpy.ndarray, numpy.ndarray. 基尔霍夫矩阵，大小为(n^2+4n)*(n^2+4n)；响应矩阵，大小为4n*4n。
    """

    """基本定义"""
    n = conductance_row.shape[0]  # 网格维数，n*n
    base = n + 2  # 基数，网格坐标为 n+2 进制的数
    coordinate = np.array(range(base ** 2)).reshape(base, base)  # 坐标序号对应矩阵

    """初始化基尔霍夫矩阵并赋值"""
    kirchhoff = np.zeros((base ** 2, base ** 2))
    for i in range(n):
        for j in range(base - 1):
            k1 = coordinate[i + 1, j]
            kirchhoff[k1, k1 + 1] = kirchhoff[k1 + 1, k1] = - conductance_row[i, j]
            k2 = coordinate[j, i + 1]
            kirchhoff[k2, k2 + base] = kirchhoff[k2 + base, k2] = - conductance_column[j, i]

    """为了方便计算响应矩阵，将点分为边界点和内部点，并重新排列"""
    boundary1 = list(coordinate[1: - 1, 0])
    boundary2 = list(coordinate[- 1, 1: - 1])
    boundary3 = list(coordinate[- 2: 0:- 1, - 1])
    boundary4 = list(coordinate[0, - 2: 0: -1])
    boundary = boundary1 + boundary2 + boundary3 + boundary4
    interior = list(coordinate[1: n + 1, 1: n + 1].reshape(n ** 2))
    rearrange = boundary + interior
    kirchhoff = kirchhoff[np.ix_(rearrange, rearrange)]

    """对角线为行(列)和"""
    kirchhoff[np.diag_indices_from(kirchhoff)] = - np.sum(kirchhoff, axis=1)

    """给定基尔霍夫矩阵，计算响应矩阵"""
    num_node = n * (n + 4)
    num_boundary = len(boundary)
    kirchhoff_boundary_to_boundary = kirchhoff[0: num_boundary, 0: num_boundary]
    kirchhoff_boundary_to_interior = kirchhoff[0: num_boundary, num_boundary: num_node]
    kirchhoff_interior_to_interior = kirchhoff[num_boundary: num_node, num_boundary: num_node]
    kirchhoff_interior_to_interior_inverse = np.linalg.inv(kirchhoff_interior_to_interior)
    response = kirchhoff_boundary_to_boundary - np.dot(kirchhoff_boundary_to_interior,
                                                       np.dot(kirchhoff_interior_to_interior_inverse,
                                                              kirchhoff_boundary_to_interior.T))

    return kirchhoff, response


def conductance_to_kirchhoff_to_response_torch(conductance_row, conductance_column):
    """
    根据给定的电导，计算基尔霍夫矩阵和响应矩阵，GPU版本。电阻网络的规模（n*n）由给定的电导（n*(n+1)和(n+1)*n）决定。
    :param conductance_row: torch.Tensor. 行电导，大小为n*(n+1)。
    :param conductance_column: torch.Tensor. 列电导，大小为(n+1)*n。
    :return: torch.Tensor, torch.Tensor. 基尔霍夫矩阵，大小为(n^2+4n)*(n^2+4n)；响应矩阵，大小为4n*4n。
    """

    """基本定义"""
    n = conductance_row.shape[0]  # 网格维数，n*n
    base = n + 2  # 基数，网格坐标为 n+2 进制的数
    coordinate = np.array(range(base ** 2)).reshape(base, base)  # 坐标序号对应矩阵

    """初始化基尔霍夫矩阵并赋值"""
    kirchhoff = torch.zeros(base ** 2, base ** 2, device=torch.device('cuda') if torch.cuda.is_available() else 'cpu')
    index_row_1_1, index_row_1_2, index_row_2_1, index_row_2_2 = [], [], [], []
    index_col_1_1, index_col_1_2, index_col_2_1, index_col_2_2 = [], [], [], []

    for i in range(n):
        for j in range(base - 1):
            k1 = coordinate[i + 1, j]
            index_row_1_1.append(k1)
            index_row_1_2.append(k1 + 1)
            index_row_2_1.append(k1 + 1)
            index_row_2_2.append(k1)

            k2 = coordinate[j, i + 1]
            index_col_1_1.append(k2)
            index_col_1_2.append(k2 + base)
            index_col_2_1.append(k2 + base)
            index_col_2_2.append(k2)

    kirchhoff[index_row_1_1, index_row_1_2] = - torch.reshape(conductance_row, (-1,))
    kirchhoff[index_row_2_1, index_row_2_2] = - torch.reshape(conductance_row, (-1,))
    kirchhoff[index_col_1_1, index_col_1_2] = - torch.reshape(conductance_column.t(), (-1,))
    kirchhoff[index_col_2_1, index_col_2_2] = - torch.reshape(conductance_column.t(), (-1,))

    """为了方便计算响应矩阵，将点分为边界点和内部点，并重新排列"""
    boundary1 = list(coordinate[1: - 1, 0])
    boundary2 = list(coordinate[- 1, 1: - 1])
    boundary3 = list(coordinate[- 2: 0:- 1, - 1])
    boundary4 = list(coordinate[0, - 2: 0: -1])
    boundary = boundary1 + boundary2 + boundary3 + boundary4
    interior = list(coordinate[1: n + 1, 1: n + 1].reshape(n ** 2))
    rearrange = boundary + interior
    kirchhoff = kirchhoff[np.ix_(rearrange, rearrange)]

    """对角线为行(列)和"""
    kirchhoff[np.diag_indices_from(kirchhoff)] = - torch.sum(kirchhoff, 1)

    """给定基尔霍夫矩阵，计算响应矩阵"""
    num_node = n * (n + 4)
    num_boundary = len(boundary)
    kirchhoff_boundary_to_boundary = kirchhoff[0: num_boundary, 0: num_boundary]
    kirchhoff_boundary_to_interior = kirchhoff[0: num_boundary, num_boundary: num_node]
    kirchhoff_interior_to_interior = kirchhoff[num_boundary: num_node, num_boundary: num_node]
    kirchhoff_interior_to_interior_inverse = torch.linalg.inv(kirchhoff_interior_to_interior)
    response = kirchhoff_boundary_to_boundary - torch.mm(kirchhoff_boundary_to_interior,
                                                         torch.mm(kirchhoff_interior_to_interior_inverse,
                                                                  kirchhoff_boundary_to_interior.t()))

    return kirchhoff, response


"""n*n网络，反问题"""


def response_to_conductance(response):
    """
    根据给定的响应矩阵，计算电导。基本算法。
    :param response: numpy.ndarray. 响应矩阵，大小为4n*4n。
    :return: numpy.ndarray, numpy.ndarray. 行电导，大小为n*(n+1)；列电导，大小为(n+1)*n。
    """

    """基本定义"""
    n = int(response.shape[0] / 4)  # 网络维数
    n1 = 3 * n
    n2 = n1 - 1
    n3 = 4 * n - 1

    """初始化电导计算值矩阵"""
    conductance_row = np.zeros([n, n + 1])
    conductance_column = np.zeros([n + 1, n])

    """对边界点顺序的置换"""
    permutation = np.append(np.arange(2 * n, 4 * n), np.arange(0, 2 * n))

    """计算电导值左上一半和右下一半"""
    for m in range(2):

        """每一半计算n层电导"""
        for i in range(n):

            """计算边界未知电势"""
            potential_boundary_unknown = - np.dot(np.linalg.inv(response[n2 - i: n1, 0: i + 1]),
                                                  response[n2 - i: n1, n3 - i])

            """计算边界未知电流，从所有非0电势位置，到待求电流位置"""
            position = np.append(np.arange(0, i + 1), [n3 - i])
            current_column = np.dot(response[np.ix_(range(n3, n3 - i, - 1), position)],
                                    np.append(potential_boundary_unknown, 1))
            current_row = np.dot(response[np.ix_(range(1, i), position)], np.append(potential_boundary_unknown, 1))
            current_flow = np.dot(response[i, position], np.append(potential_boundary_unknown, 1))
            current_flow_end = np.dot(response[n3 - i, position], np.append(potential_boundary_unknown, 1))

            """初始化电势，计算电导的节点电势，行电流算出的内部点电势，每算一层更新一次"""
            potential_start = potential_boundary_unknown[- 1]
            potential_row = potential_boundary_unknown[1: i] - current_row / conductance_row[1: i, 0]

            """列电流算出的内部点电势"""
            potential_column = - current_column / conductance_column[0, 0: i]

            for j in range(i):
                conductance_row[i - j, j] = current_flow / potential_start
                potential_temp = np.append(potential_column[j], potential_row)
                potential_end = potential_start = potential_temp[- 1]
                conductance_column[i - j, j] = - current_flow / potential_end
                if j is not (i - 1):
                    potential_temp_diff = potential_temp[0: - 1] - potential_row
                    current_temp = potential_temp_diff * conductance_column[1: i - j, j]
                    current_flow = current_flow + current_row[- 1] + current_temp[- 1]
                    current_temp_diff = current_temp[0: - 1] - current_temp[1:]
                    current_row = current_row[0: - 1] + current_temp_diff
                    potential_row = potential_row[0: - 1] - current_row / conductance_row[1: i - j - 1, j + 1]

            conductance_row[0, i] = - current_flow_end / potential_start
            conductance_column[0, i] = current_flow_end

        """交换响应矩阵的行和列，并旋转电导计算值矩阵180度"""
        response = response[permutation, :]
        response = response[:, permutation]
        conductance_row = np.rot90(conductance_row, 2)
        conductance_column = np.rot90(conductance_column, 2)

    return conductance_row, conductance_column


def response_to_conductance_sub(response):
    """
    根据给定的响应矩阵，计算电导。子图算法。
    :param response: numpy.ndarray. 响应矩阵，大小为4n*4n。
    :return: numpy.ndarray, numpy.ndarray. 行电导，大小为n*(n+1)；列电导，大小为(n+1)*n。
    """

    """基本定义"""
    n = int(response.shape[0] / 4)  # 网络维数
    base = n + 2  # 基数，网格坐标为 n+2 进制的数
    num_node_extra = base ** 2  # 所有节点个数，包括额外的四个角
    coordinate = np.array(range(num_node_extra)).reshape(base, base)  # 坐标序号对应矩阵
    kirchhoff_extend = np.zeros((num_node_extra, num_node_extra))  # 扩展的基尔霍夫矩阵初始化，包括额外的四个角

    """内点分组，左上ul，右下lr"""
    group_interior_ul = [[]]
    group_interior_lr = [[]]
    for i in range(n):
        group_interior_ul.append(coordinate[range(1, 2 + i), range(1 + i, 0, - 1)])
        group_interior_lr.append(coordinate[range(n, n - i - 1, - 1), range(n - i, n + 1)])
    group_interior = [group_interior_ul, group_interior_lr]

    """节点分层，左上ul，右下lr"""
    layer_ul = []
    layer_lr = []
    for i in range(1, n + 1):
        layer_ul.append(coordinate[0, i])
        layer_lr.append(coordinate[base - 1, base - 1 - i])
        for j in range(1, i + 1):
            layer_ul[- 1] = np.append(layer_ul[- 1], coordinate[j, i - j: i - j + 2][::- 1])
            layer_lr[- 1] = np.append(layer_lr[- 1], coordinate[base - 1 - j, base - i + j - 2: base - i + j])
    layer = [layer_ul, layer_lr]

    """用得到的序号"""
    n1 = 3 * n
    n2 = n1 - 1
    n3 = 4 * n - 1

    """对边界点顺序的置换"""
    permutation = np.append(np.arange(2 * n, 4 * n), np.arange(0, 2 * n))

    """计算电导值左上一半和右下一半"""
    for m in range(2):

        """初始化子图内点"""
        sub_interior = np.array([])

        """每一半计算n层电导"""
        for i in range(n):

            """计算边界未知电势"""
            potential_boundary_unknown = - np.dot(np.linalg.inv(response[n2 - i: n1, 0: i + 1]),
                                                  response[n2 - i: n1, n3 - i])

            """计算边界未知电流，从所有非0电势位置，到待求电流位置"""
            position = np.append(np.arange(0, i + 1), [n3 - i])
            current_boundary_unknown = np.dot(response[np.ix_(range(0, i), position)],
                                              np.append(potential_boundary_unknown, 1))
            current_flow = np.dot(response[n3 - i, position], np.append(potential_boundary_unknown, 1))

            """初始化节点电势序列，以及电流更新序列"""
            potential_sequence = np.array([1, potential_boundary_unknown[- 1]])
            current_update = np.array([0])

            """计算子图的基尔霍夫矩阵和响应矩阵"""
            if i != 0:

                """子图的边界点，内点，基尔霍夫矩阵"""
                sub_boundary = np.concatenate((coordinate[1: i + 1, 0], group_interior[m][i], coordinate[0, 1: i + 1]))
                sub_interior = np.append(sub_interior, group_interior[m][i - 1])
                sub = np.append(sub_boundary, sub_interior).astype(int)
                num_sub = len(sub)
                kirchhoff_sub = kirchhoff_extend[np.ix_(sub, sub)]
                kirchhoff_sub[range(num_sub), range(num_sub)] = - np.sum(kirchhoff_sub, axis=1)

                """子图的响应矩阵"""
                response_sub = kirchhoff_sub
                if i != 1:
                    num_sub_boundary = len(sub_boundary)
                    kirchhoff_sub_boundary_to_boundary = kirchhoff_sub[0: num_sub_boundary, 0: num_sub_boundary]
                    kirchhoff_sub_boundary_to_interior = kirchhoff_sub[0: num_sub_boundary, num_sub_boundary: num_sub]
                    kirchhoff_sub_interior_to_interior = kirchhoff_sub[
                                                         num_sub_boundary: num_sub, num_sub_boundary: num_sub
                    ]
                    kirchhoff_sub_interior_to_interior_inverse = np.linalg.inv(kirchhoff_sub_interior_to_interior)
                    temp = np.dot(kirchhoff_sub_interior_to_interior_inverse, kirchhoff_sub_boundary_to_interior.T)
                    response_sub = kirchhoff_sub_boundary_to_boundary - np.dot(kirchhoff_sub_boundary_to_interior, temp)

                """计算子图边界未知电势电流"""
                potential_sub_boundary_known = np.dot(np.linalg.inv(response_sub[0: i, i: 2 * i]),
                                                      current_boundary_unknown - np.dot(response_sub[0: i, 0: i],
                                                                                        potential_boundary_unknown[
                                                                                        0: - 1]))
                current_sub_boundary_unknown = np.dot(response_sub[i: 2 * i, 0: 2 * i],
                                                      np.append(potential_boundary_unknown[0: - 1],
                                                                potential_sub_boundary_known))

                """写出节点电势序列，以及电流更新序列"""
                potential_sequence = np.insert(potential_sequence, 1, potential_sub_boundary_known)
                current_update = np.append(current_update, current_sub_boundary_unknown)

            """给扩展的基尔霍夫矩阵赋值"""
            order_node = layer[m][i]
            for j in range(i + 1):
                current_flow = current_flow - current_update[j]
                j2 = j * 2
                kirchhoff_extend[[order_node[j2], order_node[j2 + 1]],
                                 [order_node[j2 + 1], order_node[j2]]] = - current_flow / potential_sequence[j]
                kirchhoff_extend[[order_node[j2 + 1], order_node[j2 + 2]],
                                 [order_node[j2 + 2], order_node[j2 + 1]]] = current_flow / potential_sequence[j + 1]

        """交换响应矩阵的行和列，并旋转坐标序号对应矩阵180度"""
        response = response[permutation, :]
        response = response[:, permutation]
        coordinate = np.rot90(coordinate, 2)

    """扩展的基尔霍夫矩阵转成电导行列矩阵"""
    conductance_row = np.zeros([n, n + 1])
    conductance_column = np.zeros([n + 1, n])
    for i in range(n + 1):
        conductance_row[:, i] = - kirchhoff_extend[coordinate[1: - 1, i], coordinate[1: - 1, i + 1]]
        conductance_column[i, :] = - kirchhoff_extend[coordinate[i, 1: - 1], coordinate[i + 1, 1: - 1]]

    return conductance_row, conductance_column


def response_to_conductance_improved(response):
    """
    根据给定的响应矩阵，计算电导。基本改进算法。
    :param response: numpy.ndarray. 响应矩阵，大小为4n*4n。
    :return: numpy.ndarray, numpy.ndarray. 行电导，大小为n*(n+1)；列电导，大小为(n+1)*n。
    """

    """基本定义"""
    n = int(response.shape[0] / 4)  # 网络维数
    n1 = 3 * n
    n2 = n1 - 1
    n3 = 4 * n - 1

    """初始化电导计算值矩阵"""
    conductance_row_calculate = np.zeros([n, n + 1])
    conductance_column_calculate = np.zeros([n + 1, n])

    """对边界点顺序的置换"""
    permutation = np.append(np.arange(2 * n, 4 * n), np.arange(0, 2 * n))

    """计算电导值左上一半和右下一半"""
    for m in range(2):

        """每一半计算n层电导"""
        for i in range(n):

            """计算边界未知电势"""
            potential_boundary_unknown = - np.dot(
                np.linalg.inv(response[n2 - i: n1, 0: i + 1]),
                response[n2 - i: n1, n3 - i]
            )

            """计算边界未知电流，从所有非0电势位置，到待求电流"""
            position = np.append(np.arange(0, i + 1), [n3 - i])
            current_left = np.dot(
                response[np.ix_(range(0, i), position)],
                np.append(potential_boundary_unknown, 1)
            )
            current_up = np.dot(
                response[np.ix_(range(n3, n3 - i, - 1), position)],
                np.append(potential_boundary_unknown, 1)
            )
            current_flow_left = np.dot(
                response[i, position],
                np.append(potential_boundary_unknown, 1)
            )
            current_flow_up = np.dot(
                response[n3 - i, position],
                np.append(potential_boundary_unknown, 1)
            )

            """计算内部节点电势"""
            potential_left = potential_boundary_unknown[0: i] - current_left / conductance_row_calculate[0: i, 0]
            potential_up = - current_up / conductance_column_calculate[0, 0: i]
            potential_left_start = potential_boundary_unknown[- 1]
            potential_up_start = 1

            for j in range((i+1)//2):

                """从左边计算"""
                conductance_row_calculate[i-j, j] = current_flow_left / potential_left_start
                potential_left_end = potential_left_start = potential_left[-1]
                conductance_column_calculate[i-j, j] = - current_flow_left / potential_left_end

                """从上边计算"""
                conductance_column_calculate[j, i-j] = current_flow_up / potential_up_start
                potential_up_end = potential_up_start = potential_up[-1]
                conductance_row_calculate[j, i-j] = - current_flow_up / potential_up_end

                if j is not ((i+1)//2-1):

                    """从左边计算"""
                    potential_left_diff = potential_left[0: -1] - potential_left[1:]
                    current_left_temp = potential_left_diff * conductance_column_calculate[1+j: i-j, j]
                    current_flow_left = current_flow_left + current_left[-1] + current_left_temp[-1]
                    current_left = current_left_temp[0: -1] - current_left_temp[1:] + current_left[1: -1]
                    potential_left = potential_left[1: -1] - current_left / conductance_row_calculate[1+j: i-1-j, j+1]

                    """从上边计算"""
                    potential_up_diff = potential_up[0: -1] - potential_up[1:]
                    current_up_temp = potential_up_diff * conductance_row_calculate[j, 1+j: i-j]
                    current_flow_up = current_flow_up + current_up[-1] + current_up_temp[-1]
                    current_up = current_up_temp[0: -1] - current_up_temp[1:] + current_up[1: -1]
                    potential_up = potential_up[1: -1] - current_up / conductance_column_calculate[j+1, 1+j: i-1-j]

            if (i % 2) == 0:

                k = int(i / 2)

                if i != 0:

                    """从左边计算"""
                    potential_left_diff = potential_left[0] - potential_left[-1]
                    current_left_temp = potential_left_diff * conductance_column_calculate[k, k-1]
                    current_flow_left = current_flow_left + current_left[-1] + current_left_temp

                    """从上边计算"""
                    potential_up_diff = potential_up[0] - potential_up[-1]
                    current_up_temp = potential_up_diff * conductance_row_calculate[k-1, k]
                    current_flow_up = current_flow_up + current_up[-1] + current_up_temp

                """从左边计算"""
                conductance_row_calculate[k, k] = current_flow_left / potential_left_start

                """从上边计算"""
                conductance_column_calculate[k, k] = current_flow_up / potential_up_start

        """交换响应矩阵的行和列，并旋转电导计算值矩阵180度"""
        response = response[permutation, :]
        response = response[:, permutation]
        conductance_row_calculate = np.rot90(conductance_row_calculate, 2)
        conductance_column_calculate = np.rot90(conductance_column_calculate, 2)

    return conductance_row_calculate, conductance_column_calculate


def response_to_conductance_optimize(response):
    """
    根据给定的响应矩阵，计算电导。优化算法。
    :param response: numpy.ndarray. 响应矩阵，大小为4n*4n。
    :return: numpy.ndarray, numpy.ndarray. 行电导，大小为n*(n+1)；列电导，大小为(n+1)*n。
    """

    """基本定义"""
    n = int(response.shape[0] / 4)  # 网络维数
    m = n * (n + 1)  # 电阻个数

    def vector_to_matrix(v):
        """
        根据给定电导向量，写出行电导和列电导。
        :param v: numpy.ndarray. 电阻向量，大小为2*m。
        :return: numpy.float64, numpy.float64. 行电导，大小为n*(n+1)；列电导，大小为(n+1)*n。
        """

        nonlocal n, m
        ma = v[:m].reshape(n, n + 1)
        mb = v[m:].reshape(n + 1, n)

        return ma, mb

    def vector_to_response(conductance):
        """
        根据给定电阻向量，计算响应矩阵。
        :param conductance: numpy.ndarray. 电阻向量，大小为2*m。
        :return: numpy.float64. 损失函数值，两矩阵之差的Frobenius范数平方。
        """
        nonlocal response, n, m
        conductance_row_tmp, conductance_column_tmp = vector_to_matrix(conductance)
        _, response_cal = conductance_to_kirchhoff_to_response(conductance_row_tmp, conductance_column_tmp)
        response_diff = response - response_cal  # 矩阵之差
        loss = np.linalg.norm(response_diff) ** 2  # 损失为Frobenius范数平方

        return loss

    """优化"""
    x0 = 1 / (np.ones(2 * m) * 25)  # 初始解
    bnds = tuple([(1 / 30, 1 / 20) for _ in range(2 * m)])  # 边界
    conductance_opt = minimize(vector_to_response, x0, method='L-BFGS-B', bounds=bnds)  # 优化计算
    conductance_row, conductance_column = vector_to_matrix(conductance_opt.x)

    return conductance_row, conductance_column


def response_to_conductance_optimize_torch(response):
    """
    根据给定的响应矩阵，计算电导。优化算法GPU版本。
    :param response: numpy.ndarray. 响应矩阵，大小为4n*4n。
    :return: numpy.ndarray, numpy.ndarray. 行电导，大小为n*(n+1)；列电导，大小为(n+1)*n。
    """

    """基本定义"""
    n = int(response.shape[0] / 4)  # 网络维数
    m = n * (n + 1)  # 电阻个数

    """numpy转torch"""
    response = torch.tensor(response, device=torch.device('cuda') if torch.cuda.is_available() else 'cpu')

    def vector_to_matrix(v):
        """
        根据给定电导向量，写出行电导和列电导。
        :param v: numpy.ndarray. 电导向量，大小为2*m。
        :return: numpy.float64, numpy.float64. 行电导，大小为n*(n+1)；列电导，大小为(n+1)*n。
        """

        nonlocal n, m
        ma = v[:m].reshape(n, n + 1)
        mb = v[m:].reshape(n + 1, n)

        return ma, mb

    def vector_to_response(conductance):
        """
        根据给定电阻向量，计算响应矩阵。
        :param conductance: torch.Tensor. 电导向量，大小为2*m。
        :return: torch.Tensor. 损失函数值，两矩阵之差的Frobenius范数平方。
        """

        nonlocal response, n, m
        conductance_row_tmp, conductance_column_tmp = vector_to_matrix(conductance)
        _, response_cal = conductance_to_kirchhoff_to_response_torch(conductance_row_tmp, conductance_column_tmp)
        response_diff = response - response_cal  # 矩阵之差
        loss = torch.norm(response_diff) ** 2  # 损失为Frobenius范数平方

        return loss

    """优化"""
    x0 = 1 / (torch.ones([2 * m], device=torch.device('cuda') if torch.cuda.is_available() else 'cpu') * 25)  # 初始解
    x0.requires_grad = True  # 可求梯度
    optimizer = torch.optim.Adam([x0], lr=1e-3)  # 优化器
    for _ in range(1000):
        y = vector_to_response(x0)
        optimizer.zero_grad()
        y.backward()
        optimizer.step()
        if y <= 1e-10:
            break
    conductance_row, conductance_column = vector_to_matrix(x0.detach().cpu().numpy())

    return conductance_row, conductance_column


"""n*n网络，还原误差"""


def resistance_diff(conductance_a_row, conductance_a_col, conductance_b_row, conductance_b_col):
    """
    计算电阻向量之差。
    :param conductance_a_row: numpy.ndarray. 行电导a，大小为n*(n+1)。
    :param conductance_a_col: numpy.ndarray. 列电导a，大小为(n+1)*n。
    :param conductance_b_row: numpy.ndarray. 行电导b，大小为n*(n+1)。
    :param conductance_b_col: numpy.ndarray. 列电导b，大小为(n+1)*n。
    :return: numpy.float64, numpy.float64, numpy.float64. 矩阵之差绝对值的总和，最大值，均值。
    """

    conductance_a = np.concatenate((np.ravel(conductance_a_row), np.ravel(conductance_a_col)))
    conductance_b = np.concatenate((np.ravel(conductance_b_row), np.ravel(conductance_b_col)))
    diff = np.abs(1 / conductance_a - 1 / conductance_b)
    diff_sum = np.sum(diff)
    diff_max = np.max(diff)
    diff_mean = np.mean(diff)

    return diff_sum, diff_max, diff_mean


"""自定义网络，正问题"""


def conductance_to_kirchhoff_to_response_customize(num_boundary, num_interior, edge, conductance):
    """
    根据自定义的顶点、边和电导，计算基尔霍夫矩阵和响应矩阵。
    :param num_boundary: int. 边界点个数。
    :param num_interior: int. 内点个数.
    :param edge: list. 边拓扑。
    :param conductance: list. 边电导。
    :return: numpy.ndarray, numpy.ndarray. 基尔霍夫矩阵，大小为(num_boundary+num_interior)*(num_boundary+num_interior)；
    响应矩阵，大小为num_boundary*num_boundary。
    """

    num_node = num_boundary + num_interior  # 顶点个数

    """计算基尔霍夫矩阵"""
    kirchhoff = np.zeros((num_node, num_node))
    for i in range(len(edge)):
        j, k = edge[i]
        kirchhoff[j, k] = kirchhoff[k, j] = - conductance[i]
    kirchhoff[np.diag_indices_from(kirchhoff)] = - np.sum(kirchhoff, axis=1)

    """计算响应矩阵"""
    kirchhoff_boundary_to_boundary = kirchhoff[0: num_boundary, 0: num_boundary]
    kirchhoff_boundary_to_interior = kirchhoff[0: num_boundary, num_boundary: num_node]
    kirchhoff_interior_to_interior = kirchhoff[num_boundary: num_node, num_boundary: num_node]
    kirchhoff_interior_to_interior_inverse = np.linalg.inv(kirchhoff_interior_to_interior)
    response = kirchhoff_boundary_to_boundary - np.dot(kirchhoff_boundary_to_interior,
                                                       np.dot(kirchhoff_interior_to_interior_inverse,
                                                              kirchhoff_boundary_to_interior.T))

    return kirchhoff, response


def conductance_to_kirchhoff_to_response_customize_torch(num_boundary, num_interior, edge, conductance):
    """
    根据自定义的顶点、边和电导，计算基尔霍夫矩阵和响应矩阵，GPU版本。
    :param num_boundary: int. 边界点个数。
    :param num_interior: int. 内点个数.
    :param edge: list. 边拓扑。
    :param conductance: torch.Tensor. 边电导。
    :return: torch.Tensor, torch.Tensor. 基尔霍夫矩阵，大小为(num_boundary+num_interior)*(num_boundary+num_interior)；
    响应矩阵，大小为num_boundary*num_boundary。
    """

    num_node = num_boundary + num_interior  # 顶点个数

    """计算基尔霍夫矩阵"""
    kirchhoff = torch.zeros(num_node, num_node, device=torch.device('cuda') if torch.cuda.is_available() else 'cpu')
    index_1 = list(map(lambda x: x[0], edge))
    index_2 = list(map(lambda x: x[1], edge))

    kirchhoff[index_1, index_2] = - conductance
    kirchhoff[index_2, index_1] = - conductance

    kirchhoff[np.diag_indices_from(kirchhoff)] = - torch.sum(kirchhoff, 1)

    """计算响应矩阵"""
    kirchhoff_boundary_to_boundary = kirchhoff[0: num_boundary, 0: num_boundary]
    kirchhoff_boundary_to_interior = kirchhoff[0: num_boundary, num_boundary: num_node]
    kirchhoff_interior_to_interior = kirchhoff[num_boundary: num_node, num_boundary: num_node]
    kirchhoff_interior_to_interior_inverse = torch.linalg.inv(kirchhoff_interior_to_interior)
    response = kirchhoff_boundary_to_boundary - torch.mm(kirchhoff_boundary_to_interior,
                                                         torch.mm(kirchhoff_interior_to_interior_inverse,
                                                                  kirchhoff_boundary_to_interior.t()))

    return kirchhoff, response


"""自定义网络，反问题"""


def response_to_conductance_customize_optimize_torch(num_boundary, num_interior, edge, response):
    """
    根据自定义的顶点、边和电导，以及给定的响应矩阵，计算电导。优化算法GPU版本。
    :param num_boundary: int. 边界点个数。
    :param num_interior: int. 内点个数。
    :param edge: list. 边拓扑。
    :param response: numpy.ndarray. 响应矩阵。
    :return: numpy.ndarray. 电导
    """

    """numpy转torch"""
    response = torch.tensor(response, device=torch.device('cuda') if torch.cuda.is_available() else 'cpu')

    def vector_to_response(conductance):
        """
        根据给定电阻向量，计算响应矩阵。
        :param conductance: torch.Tensor. 电导向量。
        :return: torch.Tensor. 损失函数值，两矩阵之差的Frobenius范数平方。
        """

        nonlocal num_boundary, num_interior, edge, response
        _, response_cal = conductance_to_kirchhoff_to_response_customize_torch(
            num_boundary, num_interior, edge, conductance
        )
        response_diff = response - response_cal  # 矩阵之差
        loss = torch.norm(response_diff) ** 2  # 损失为Frobenius范数平方

        return loss

    """优化"""
    x0 = 1 / (torch.ones([len(edge)], device=torch.device('cuda') if torch.cuda.is_available() else 'cpu') * 25)  # 初始解
    x0.requires_grad = True  # 可求梯度
    optimizer = torch.optim.Adam([x0], lr=1e-3)  # 优化器
    for _ in range(1000):
        y = vector_to_response(x0)
        optimizer.zero_grad()
        y.backward()
        optimizer.step()
        # if y <= 1e-10:
        #     break

    return x0.detach().cpu().numpy()


"""自定义网络，还原误差"""


def resistance_diff_customize(conductance_a, conductance_b):
    """
    计算电阻向量之差。
    :param conductance_a: numpy.ndarray. 电导a。
    :param conductance_b: numpy.ndarray. 电导b。
    :return: numpy.float64, numpy.float64, numpy.float64. 向量之差绝对值的总和，最大值，均值。
    """

    diff = np.abs(1 / conductance_a - 1 / conductance_b)
    diff_sum = np.sum(diff)
    diff_max = np.max(diff)
    diff_mean = np.mean(diff)

    return diff_sum, diff_max, diff_mean


if __name__ == '__main__':

    """n*n网络"""
    """随机生成电导"""
    # row = 1 / np.random.uniform(20, 30, (4, 5))
    # col = 1 / np.random.uniform(20, 30, (5, 4))
    # _, re = conductance_to_kirchhoff_to_response(row, col)
    # re = re + np.random.normal(0, 0.000000001, (16, 16))
    """依次测试还原算法"""
    # row_cal, col_cal = response_to_conductance(re)  # 基本算法
    # print(resistance_diff(row, col, row_cal, col_cal))
    # row_cal, col_cal = response_to_conductance_sub(re)  # 子图算法
    # print(resistance_diff(row, col, row_cal, col_cal))
    # row_cal, col_cal = response_to_conductance_improved(re)  # 改进基本算法
    # print(resistance_diff(row, col, row_cal, col_cal))
    # row_cal, col_cal = response_to_conductance_optimize(re)  # 优化算法
    # print(resistance_diff(row, col, row_cal, col_cal))

    """六芒星网络"""
    # connect = [
    #     [0, 1], [1, 2], [2, 10], [10, 3], [3, 4], [4, 5],
    #     [5, 11], [11, 6], [6, 7], [7, 8], [8, 9], [9, 0],
    #     [1, 10], [10, 4], [4, 11], [11, 7], [7, 9], [9, 1],
    #     [1, 12], [10, 12], [4, 12], [11, 12], [7, 12], [9, 12]
    # ]
    """随机生成电导"""
    # con = 1 / np.random.uniform(20, 30, 24)
    # _, re = conductance_to_kirchhoff_to_response_customize(9, 4, connect, con)
    """测试还原算法"""
    # con_cal = response_to_conductance_customize_optimize_torch(9, 4, connect, re)
    # print(resistance_diff_customize(con, con_cal))
