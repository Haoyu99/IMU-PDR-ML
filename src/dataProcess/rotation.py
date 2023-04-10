import numpy as np
import open3d as o3d
import copy

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=20, origin=[-2, -2, -2])
pcd = o3d.io.read_point_cloud("iPhonex.pcd")



vis = o3d.visualization.Visualizer()
vis.create_window(window_name='pcd', width=1440, height=1080)
vis.add_geometry(pcd)
vis.add_geometry(mesh_frame)
vis.get_render_option().load_from_json('renderoption.json')
vis.run()  # user changes the view and press "q" to terminate
param = vis.get_view_control().convert_to_pinhole_camera_parameters()
o3d.io.write_pinhole_camera_parameters(pcd, "D:\DataSet\RIDI\Projetct\source\draw.json")





#
# -------------------------------------
# 直接位移
# pcd = pcd.translate((20, 0, 0))
# -------------------------------------
# 通过欧拉角位移
# th = np.array([0, np.pi/3, 0]).T
# thAxis = pcd.get_rotation_matrix_from_axis_angle(th)
# pcd.rotate(thAxis)

# -------------------------------------
# 通过四元数旋转
#0.683728 0.290054 0.228332 0.629483  bag
quart = np.array([0.683746, 0.290139, 0.228645 ,0.629312]).T
thQuart = pcd.get_rotation_matrix_from_quaternion(quart)
pcd.rotate(thQuart)
#
 # 把float数组转换成四元数对象
# quart = np.array([0.683746, -0.290139, -0.228645, -0.629312]).T
# thQuart = pcd.get_rotation_matrix_from_quaternion(quart)
# pcd.rotate(thQuart)
# o3d.visualization.draw_geometries([mesh_frame, pcd])


