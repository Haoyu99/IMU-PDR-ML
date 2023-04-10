import open3d as o3d
#模型路径，支持后缀：stl/ply/obj/off/gltf/glb
path_obj = "D:\DataSet\RIDI\\iphonex.ply"
#读入网格模型
mesh = o3d.io.read_triangle_mesh(path_obj)
#计算网格顶点
mesh.compute_vertex_normals()
#可视化网格模型
o3d.visualization.draw_geometries([mesh])

# #均匀采样5000个点
# pcd = mesh.sample_points_uniformly(number_of_points=10000)
# #可视化点云模型
# o3d.visualization.draw_geometries([pcd])

#poisson_disk方法采样5000个点
pcd = mesh.sample_points_poisson_disk(number_of_points=10000, init_factor=10)
#可视化点云模型
o3d.visualization.draw_geometries([pcd])
#保存
o3d.io.write_point_cloud("iPhonex.pcd", pcd)
