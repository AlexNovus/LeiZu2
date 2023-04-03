import numpy as np
import scipy.io as io
import openmesh as om
import igl
import matplotlib.pyplot as plt


model = np.load('basic_model.npy')
face = io.loadmat('testFaces.mat')['faces'] - 1

mesh = om.TriMesh()
mesh.add_vertices(model)
mesh.add_faces(face)
face = mesh.fv_indices()
edge = mesh.ev_indices()

"""边界点序号"""
boundary = igl.boundary_loop(face)

"""边界点映射"""
boundary_uv = igl.map_vertices_to_circle(model, boundary)

"""网格整体映射"""
uv = igl.harmonic_weights(model, face, boundary, boundary_uv, 1)
print(uv)

# v_p = np.hstack([uv, np.zeros((uv.shape[0],1))])

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 17
plt.rcParams["axes.labelpad"] = 8

"""可视化"""
fig, ax = plt.subplots(figsize=(8, 8), dpi=50)
ax.scatter(uv[:, 0], uv[:, 1], s=5)
for e in edge:
    x = uv[e, 0]
    y = uv[e, 1]
    ax.plot(x, y, c='#1f77b4', linewidth=1)
plt.show()
