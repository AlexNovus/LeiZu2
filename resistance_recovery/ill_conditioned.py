from test import *


def scale_error(func, n):
    """
    可视化误差随网络维数的变化。
    :param func: function. 还原算法函数。
    :param n: int. 网络维数。
    :return:
    """

    result = []
    for s in range(1, n + 1):
        result.append(random_test(func, s))
    result = np.array(result)
    """画图参数设置"""
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    """可视化"""
    visual(result[:, 0], '电阻平均误差（欧姆）')


def measure_error(func, n, std_inf, std_sup):
    """
    误差随测量误差的变化。
    :param func: function. 还原算法函数。
    :param n: int. 网络维数。
    :param std_inf: int. 标准差对数下界。
    :param std_sup: int. 标准差对数上界。
    :return:
    """

    res = []
    for std in range(std_inf, std_sup + 1):
        res.append(random_test_measure(func, n, 10 ** std))
    res = np.array(res)

    return res


def measure_error_customize(std_inf, std_sup):
    """
    误差随测量误差的变化。
    :param std_inf: int. 标准差对数下界。
    :param std_sup: int. 标准差对数上界。
    :return:
    """

    res = []
    edge = [
        [0, 1], [1, 2], [2, 10], [10, 3], [3, 4], [4, 5],
        [5, 11], [11, 6], [6, 7], [7, 8], [8, 9], [9, 0],
        [1, 10], [10, 4], [4, 11], [11, 7], [7, 9], [9, 1],
        [1, 12], [10, 12], [4, 12], [11, 12], [7, 12], [9, 12]
    ]
    for std in range(std_inf, std_sup + 1):
        print(std)
        res.append(random_test_customize_measure(9, 4, edge, 10 ** std))
    res = np.array(res)

    return res


def condition_number(n):
    """
    计算响应矩阵子矩阵条件数。
    :param n: int. 网络维数。
    :return: float. 条件数。
    """

    conductance_row = 1 / np.ones((n, n + 1)) * 20
    conductance_col = 1 / np.ones((n + 1, n)) * 20
    _, response = conductance_to_kirchhoff_to_response(conductance_row, conductance_col)
    sub = response[2 * n: 3 * n, 0: n]
    sub_inv = np.linalg.inv(sub)

    return np.linalg.norm(sub_inv, 2) * np.linalg.norm(sub, 2)


def visual_log(data):
    """
    还原算法性能可视化。
    :param data: 数据。
    :return:
    """

    x = list(map(lambda y: 10 ** y, range(-15, -5)))  # 横坐标
    """画图"""
    fig, ax = plt.subplots(figsize=(15, 9), dpi=50)
    ax.plot(x, data, 'o-')
    ax.set(
        xlabel='测量误差标准差',
        ylabel='电阻平均误差（欧姆）',
        xlim=(1e-16, 1e-5),
        xscale='log'
    )
    ax.set_xticks(x)
    ax.grid()
    plt.show()


def visual_log_compare(data_1, data_2, label_1, label_2):
    """
    还原算法性能对比可视化。
    :param data_1: 数据1。
    :param data_2: 数据2。
    :param label_1: str. 数据1标签。
    :param label_2: str. 数据2标签。
    :return:
    """

    x = list(map(lambda y: 10 ** y, range(-15, -5)))  # 横坐标
    """画图"""
    fig, ax = plt.subplots(figsize=(15, 9), dpi=50)
    ax.plot(x, data_1, 'o-', label=label_1)
    ax.plot(x, data_2, 'o-', label=label_2)
    ax.set(
        xlabel='测量误差标准差',
        ylabel='电阻平均误差（欧姆）',
        xlim=(1e-16, 1e-5),
        xscale='log'
    )
    ax.set_xticks(x)
    ax.grid()
    ax.legend()
    plt.show()


if __name__ == '__main__':

    """画图参数设置"""
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 27
    plt.rcParams["axes.labelpad"] = 8

    """误差随网络规模变化"""
    # scale_error(response_to_conductance_improved, 14)

    """误差随测量误差标准差变化"""
    """n*n网络"""
    """基本还原算法"""
    # res_improved_measure = measure_error(response_to_conductance_improved, 4, -15, -6)
    # np.save('result/res_improved_measure.npy', res_improved_measure)

    """基于优化的还原算法算法"""
    # res_optimize_measure = measure_error(response_to_conductance_optimize_torch, 4, -15, -6)
    # np.save('result/res_optimize_measure.npy', res_optimize_measure)

    """六芒星网络"""
    """基于优化的还原算法算法"""
    # res_customize_optimize_measure = measure_error_customize(-15, -6)
    # np.save('optimize/res_customize_optimize_measure.npy', res_customize_optimize_measure)

    """条件数随网络规模变化"""
    # cond_num = []
    # for size in range(1, 15):
    #     cond_num.append(condition_number(size))
    # print(cond_num)
    # visual(cond_num, '条件数')

    """读取数据"""
    res_improved_measure = np.load('result/res_improved_measure.npy')
    res_optimize_measure = np.load('result/res_optimize_measure.npy')
    res_customize_optimize_measure = np.load('result/res_customize_optimize_measure.npy')
    """单算法可视化"""
    # visual_log(res_customize_optimize_measure[:, 0])
    """多算法可视化"""
    visual_log_compare(res_improved_measure[:, 0], res_optimize_measure[:, 0], '基本算法', '优化算法')
