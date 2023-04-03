import time
from elecnet import *
import matplotlib.pyplot as plt


def random_test(func, n, iteration=1000):
    """
    随机生成电阻，测试电阻还原算法的精度和速度。
    :param func: function. 电阻还原函数。
    :param n: int. 网络维数。
    :param iteration: int. 随机迭代次数。
    :return: float. float. 平均运行时间。平均还原误差。
    """

    total_diff = 0
    total_time = 0
    for i in range(iteration):
        """随机生成电导，计算响应矩阵"""
        np.random.seed(i)
        conductance_row = 1 / np.random.uniform(20, 30, (n, n + 1))
        conductance_col = 1 / np.random.uniform(20, 30, (n + 1, n))
        _, response = conductance_to_kirchhoff_to_response(conductance_row, conductance_col)
        """记录时间和误差"""
        start_time = time.time()
        conductance_row_cal, conductance_col_cal = func(response)
        end_time = time.time()
        total_time += end_time - start_time
        _, _, diff_mean = resistance_diff(conductance_row, conductance_col, conductance_row_cal, conductance_col_cal)
        total_diff += diff_mean

    return [total_diff / iteration, total_time / iteration]


def random_test_measure(func, n, std, iteration=1000):
    """
    随机生成电阻，施加随机测量误差，测试电阻还原算法的精度和速度。
    :param func: 电阻还原函数。
    :param n: 网络维数。
    :param std: float. 测量误差正态分布标准差。
    :param iteration: int. 随机迭代次数。
    :return: float. float. 平均运行时间。平均还原误差。
    """

    total_diff = 0
    total_time = 0
    for i in range(iteration):
        """随机生成电导，计算响应矩阵"""
        np.random.seed(i)
        conductance_row = 1 / np.random.uniform(20, 30, (n, n + 1))
        conductance_col = 1 / np.random.uniform(20, 30, (n + 1, n))
        _, response = conductance_to_kirchhoff_to_response(conductance_row, conductance_col)
        response = response + np.random.normal(0, std, (4 * n, 4 * n))
        """记录时间和误差"""
        start_time = time.time()
        conductance_row_cal, conductance_col_cal = func(response)
        end_time = time.time()
        total_time += end_time - start_time
        _, _, diff_mean = resistance_diff(conductance_row, conductance_col, conductance_row_cal, conductance_col_cal)
        total_diff += diff_mean

    return [total_diff / iteration, total_time / iteration]


def random_test_customize(num_boundary, num_interior, edge, iteration=1000):
    """
    在自定义网络上随机生成电阻，测试电阻还原算法的精度和速度。
    :param num_boundary: int. 边界点个数。
    :param num_interior: int. 内点个数。
    :param edge: list. 边拓扑。
    :param iteration: int. 随机迭代次数。
    :return: float. float. 平均运行时间。平均还原误差。
    """

    total_diff = 0
    total_time = 0
    for i in range(iteration):
        """随机生成电导，计算响应矩阵"""
        np.random.seed(i)
        conductance = 1 / np.random.uniform(20, 30, len(edge))
        _, response = conductance_to_kirchhoff_to_response_customize(
            num_boundary, num_interior, edge, conductance
        )
        """记录时间和误差"""
        start_time = time.time()
        conductance_cal = response_to_conductance_customize_optimize_torch(num_boundary, num_interior, edge, response)
        end_time = time.time()
        total_time += end_time - start_time
        _, _, diff_mean = resistance_diff_customize(conductance, conductance_cal)
        total_diff += diff_mean

    return [total_diff / iteration, total_time / iteration]


def random_test_customize_measure(num_boundary, num_interior, edge, std, iteration=1000):
    """
    在自定义网络上随机生成电阻，施加随机测量误差，测试电阻还原算法的精度和速度。
    :param num_boundary: int. 边界点个数。
    :param num_interior: int. 内点个数。
    :param edge: list. 边拓扑。
    :param std: float. 测量误差正态分布标准差。
    :param iteration: int. 随机迭代次数。
    :return: float. float. 平均运行时间。平均还原误差。
    """

    total_diff = 0
    total_time = 0
    for i in range(iteration):
        """随机生成电导，计算响应矩阵"""
        np.random.seed(i)
        conductance = 1 / np.random.uniform(20, 30, len(edge))
        _, response = conductance_to_kirchhoff_to_response_customize(num_boundary, num_interior, edge, conductance)
        response = response + np.random.normal(0, std, (num_boundary, num_boundary))

        """记录时间和误差"""
        start_time = time.time()
        conductance_cal = response_to_conductance_customize_optimize_torch(num_boundary, num_interior, edge, response)
        end_time = time.time()
        total_time += end_time - start_time
        _, _, diff_mean = resistance_diff_customize(conductance, conductance_cal)
        total_diff += diff_mean

    return [total_diff / iteration, total_time / iteration]


def visual(data, label_y):
    """
    还原算法性能可视化。
    :param data: 数据。
    :param label_y: str. y轴标签。
    :return:
    """

    x = list(range(1, len(data) + 1))  # 横坐标
    """画图"""
    fig, ax = plt.subplots(figsize=(10, 9), dpi=50)
    # fig, ax = plt.subplots(figsize=(15, 9), dpi=50)
    ax.plot(x, data, 'o-')
    ax.set(
        xlabel='电阻网络阶数',
        ylabel=label_y,
        xlim=(0, len(data) + 1),
        xticks=x
    )
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.grid()
    plt.show()


def visual_compare(data_1, data_2, label_1, label_2, label_y):
    """
    还原算法性能对比可视化。
    :param data_1: 数据1。
    :param data_2: 数据2。
    :param label_1: str. 数据1标签。
    :param label_2: str. 数据2标签。
    :param label_y: str. y轴标签。
    :return:
    """

    x = list(range(1, len(data_1) + 1))  # 横坐标
    """画图"""
    fig, ax = plt.subplots(figsize=(15, 9), dpi=50)
    ax.plot(x, data_1, 'o-', label=label_1)
    ax.plot(x, data_2, 'o-', label=label_2)
    ax.set(
        xlabel='电阻网络阶数',
        ylabel=label_y,
        xlim=(0, len(data_1) + 1),
        xticks=x
    )
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.grid()
    ax.legend()
    plt.show()


if __name__ == '__main__':

    """n*n网络"""
    """基本还原算法"""
    # res_improved = []
    # for size in range(1, 15):
    #     res_improved.append(random_test(response_to_conductance_improved, size))
    # res_improved = np.array(res_improved)
    # np.save('result/res_improved.npy', res_improved)
    """子网络还原算法"""
    # res_sub = []
    # for size in range(1, 15):
    #     res_sub.append(random_test(response_to_conductance_sub, size))
    # res_sub = np.array(res_sub)
    # np.save('result/res_sub.npy', res_sub)
    """基于优化的还原算法"""
    # res_optimize = []
    # for size in range(1, 15):
    #     print(size)
    #     res_optimize.append(random_test(response_to_conductance_optimize_torch, size))
    # res_optimize = np.array(res_optimize)
    # np.save('result/res_optimize.npy')

    """六芒星网格"""
    """基于优化的还原算法"""
    # connect = [
    #     [0, 1], [1, 2], [2, 10], [10, 3], [3, 4], [4, 5],
    #     [5, 11], [11, 6], [6, 7], [7, 8], [8, 9], [9, 0],
    #     [1, 10], [10, 4], [4, 11], [11, 7], [7, 9], [9, 1],
    #     [1, 12], [10, 12], [4, 12], [11, 12], [7, 12], [9, 12]
    # ]
    # print(random_test_customize(9, 4, connect))  # 结果为[0.014279156887421141, 1.761558840751648]

    """画图参数设置"""
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 27
    plt.rcParams["axes.labelpad"] = 8
    """读取数据"""
    res_improved = np.load('result/res_improved.npy')
    res_sub = np.load('result/res_sub.npy')
    res_optimize = np.load('result/res_optimize.npy')
    """单算法可视化"""
    visual(res_optimize[:, 0], '电阻平均误差（欧姆）')
    visual(res_optimize[:, 1], '平均运行时间（秒）')
    """多算法可视化"""
    # visual_compare(res_improved[:, 0], res_optimize[:, 0], '基本算法', '优化算法', '电阻平均误差（欧姆）')
    # visual_compare(res_improved[:, 1], res_optimize[:, 1], '基本算法', '优化算法', '平均运行时间（秒）')
