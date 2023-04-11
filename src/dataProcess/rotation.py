import numpy as np
import open3d as o3d


# 使用四元数旋转并且展示世界坐标系下的手机状态
def rotation_and_show(quaternion):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=20, origin=[-2, -2, -2])
    phone = o3d.io.read_point_cloud("./resources/iPhonex.pcd")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=1440, height=1080)
    vis.add_geometry(phone)
    vis.add_geometry(mesh_frame)
    # quart = np.array([0.78648,0.617316,-0.018206,-0.006167]).T
    thQuart = phone.get_rotation_matrix_from_quaternion(quaternion)
    phone.rotate(thQuart)
    vis.run()


rotation_and_show([0.78648, 0.617316, -0.018206, -0.006167])
