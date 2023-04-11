import numpy as np
import open3d as o3d


def rotation_and_show(quaternion):
    """
    使用四元数旋转并且展示世界坐标系下的手机状态
    :param quaternion: 单位四元数（q0,q1,q2,q3）
    :return:
    """
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=20, origin=[-2, -2, -2])
    phone = o3d.io.read_point_cloud("./resources/iPhonex.pcd")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=1440, height=1080)
    vis.add_geometry(phone)
    vis.add_geometry(mesh_frame)
    thQuart = phone.get_rotation_matrix_from_quaternion(quaternion)
    phone.rotate(thQuart)
    vis.run()


def get_rotation_matrix_from_quaternion(quaternion):
    """
    从四元数得到当前设备的旋转矩阵，用于后续的坐标变换
    :param quaternion: 单位四元数
    :return: R: 旋转矩阵(3 * 3)
    """
    q0 = quaternion[0]
    q1 = quaternion[1]
    q2 = quaternion[2]
    q3 = quaternion[3]
    sq_q1 = 2 * q1 * q1
    sq_q2 = 2 * q2 * q2
    sq_q3 = 2 * q3 * q3
    q1_q2 = 2 * q1 * q2
    q3_q0 = 2 * q3 * q0
    q1_q3 = 2 * q1 * q3
    q2_q0 = 2 * q2 * q0
    q2_q3 = 2 * q2 * q3
    q1_q0 = 2 * q1 * q0
    R = np.zeros((3, 3))
    R[0][0] = 1 - sq_q2 - sq_q3
    R[0][1] = q1_q2 - q3_q0
    R[0][2] = q1_q3 + q2_q0
    R[1][0] = q1_q2 + q3_q0
    R[1][1] = 1 - sq_q1 - sq_q3
    R[1][2] = q2_q3 - q1_q0
    R[2][0] = q1_q3 - q2_q0
    R[2][1] = q2_q3 + q1_q0
    R[2][2] = 1 - sq_q1 - sq_q2
    return R


# 测试
# rotation_and_show([0.007253, -0.301691, -0.953353, 0.006899])
a = get_rotation_matrix_from_quaternion([0.007253, -0.301691, -0.953353, 0.006899])
