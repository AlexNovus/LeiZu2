import numpy as np


def evaluate_error(ground_truth, prediction):
    """
    衡量真实三维模型和预测三维模型的平均距离误差。共n个模型，每个模型m个顶点。
    :param ground_truth: numpy.ndarray. 真实三维模型，大小为n*(m*3)。
    :param prediction: numpy.ndarray. 预测三维模型，大小为n*(m*3)。
    :return: float. 平均距离误差。
    """

    model_diff = ground_truth - prediction
    avg_dist_error = []  # 记录每个模型的平均距离误差
    for diff in model_diff:
        dist_error = list(map(np.linalg.norm, diff.reshape(-1, 3)))
        avg_dist_error.append(np.mean(dist_error))

    return np.mean(avg_dist_error)


def evaluate_error_single(ground_truth, prediction):
    """
    衡量单个真实三维模型和预测三维模型每个顶点的距离误差。模型m个顶点。
    :param ground_truth: numpy.ndarray. 真实三维模型，大小为1*(m*3)。
    :param prediction: numpy.ndarray. 预测三维模型，大小为1*(m*3)。
    :return: numpy.ndarray. 顶点距离误差，大小为1*(m*3)。
    """

    model_diff = ground_truth - prediction
    dist_error = list(map(np.linalg.norm, model_diff.reshape(-1, 3)))

    return np.array(dist_error)


# def show_error(ground_truth, prediction, index_gt):
#     """旧函数，不使用"""
#
#     error = ground_truth - prediction
#     num_samples = np.shape(error)[0]
#     num_vertices = int(np.shape(error)[1] / 3)
#
#     average_distance_error_all = np.zeros(num_samples)  # 所有模型的平均距离误差
#     max_distance_error_all = np.zeros(num_samples)  # 所有模型的最大距离误差
#     average_relative_distance_error_all = np.zeros(num_samples)  # 所有模型的平均相对误差
#     max_relative_distance_error_all = np.zeros(num_samples)  # 所有模型的最大相对误差
#
#     for index_model in range(num_samples):
#
#         y = ground_truth[index_model].reshape((num_vertices, 3))[:, 1]
#         length = np.max(y) - np.min(y)
#
#         distance_error_one = np.zeros(num_vertices)  # 该模型所有顶点距离误差
#
#         for index_vertex in range(num_vertices):
#             l1 = error[index_model, 3 * index_vertex]
#             l2 = error[index_model, 3 * index_vertex + 1]
#             l3 = error[index_model, 3 * index_vertex + 2]
#             distance_error_one[index_vertex] = (l1 ** 2 + l2 ** 2 + l3 ** 2) ** 0.5
#
#         mean_one = np.mean(distance_error_one)
#         max_one = np.max(distance_error_one)
#         average_distance_error_all[index_model] = mean_one  # 该模型平均距离误差
#         max_distance_error_all[index_model] = max_one  # 该模型最大距离误差
#         average_relative_distance_error_all[index_model] = mean_one / length * 100  # 该模型平均相对误差
#         max_relative_distance_error_all[index_model] = max_one / length * 100  # 该模型最大相对误差
#
#     print('average distance error:', np.mean(average_distance_error_all))
#     print(index_gt[np.argmax(max_distance_error_all)], 'has max distance error:', np.max(max_distance_error_all))
#     arde = np.mean(average_relative_distance_error_all)
#     print('average relative distance error:', arde, '%')
#     print(index_gt[np.argmax(max_relative_distance_error_all)], 'has max relative distance error:',
#           np.max(max_relative_distance_error_all), '%')
#
#     return arde
