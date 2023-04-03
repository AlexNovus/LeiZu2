import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams["axes.labelpad"] = 8


# points = np.array([[0, 0], [0.9, 0.5], [1.9, 0.2], [3.1, 0.8],
#                    [0.2, 1.3], [1.3, 0.9], [2.4, 1.5],
#                    [0.1, 2.1], [1.1, 2], [2, 1.8], [3.5, 2.2],
#                    [2, -5], [10, 5], [2, 10], [-2, 5]])
# vor = Voronoi(points)
# fig = voronoi_plot_2d(vor, show_points=False, show_vertices=False)
# plt.scatter(points[:, 0], points[:, 1], color='black')
# plt.xlim(-1, 4)
# plt.ylim(-0.5, 2.7)
# pairs = [
#     [0, 1], [1, 2], [2, 3],
#     [0, 4], [1, 4], [1, 5], [2, 5], [2, 6], [3, 6],
#     [4, 5], [5, 6],
#     [4, 7], [4, 8], [5, 8], [5, 9], [6, 9], [6, 10], [3, 10],
#     [7, 8], [8, 9], [9, 10]
# ]
# for x, y in pairs:
#     plt.plot(points[[x, y], 0], points[[x, y], 1], color='black', linestyle='dashed')
# plt.show()

points = np.array([[0, 0], [1, 0], [2, 0], [3, 0],
                   [0.5, 1], [1.5, 1], [2.5, 1],
                   [0, 2], [1, 2], [2, 2], [3, 2],
                   [-1, 0], [4, 0],
                   [-1, 2], [4, 2],
                   [-0.5, 1], [3.5, 1],
                   [-0.5, -1], [0.5, -1], [1.5, -1], [2.5, -1], [3.5, -1],
                   [-0.5, 3], [0.5, 3], [1.5, 3], [2.5, 3], [3.5, 3],])
vor = Voronoi(points)
fig = voronoi_plot_2d(vor, show_points=False, show_vertices=False)
plt.scatter(points[:, 0], points[:, 1], color='black')
plt.xlim(-0.4, 3.4)
plt.ylim(-0.4, 2.4)
pairs = [
    [0, 1], [1, 2], [2, 3],
    [0, 4], [1, 4], [1, 5], [2, 5], [2, 6], [3, 6],
    [4, 5], [5, 6],
    [4, 7], [4, 8], [5, 8], [5, 9], [6, 9], [6, 10],
    [7, 8], [8, 9], [9, 10]
]
for x, y in pairs:
    plt.plot(points[[x, y], 0], points[[x, y], 1], color='black', linestyle='dashed')
plt.show()
