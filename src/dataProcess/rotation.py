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
    :param quaternion: 单位四元数 n * 4 一共有n组四元数
    :return: R: 旋转矩阵n * (3 * 3)
    """
    q0 = quaternion[:, 0]  # 从输入的四元数数组中获取 q0
    q1 = quaternion[:, 1]  # 从输入的四元数数组中获取 q1
    q2 = quaternion[:, 2]  # 从输入的四元数数组中获取 q2
    q3 = quaternion[:, 3]  # 从输入的四元数数组中获取 q3
    sq_q1 = 2 * q1 * q1
    sq_q2 = 2 * q2 * q2
    sq_q3 = 2 * q3 * q3
    q1_q2 = 2 * q1 * q2
    q3_q0 = 2 * q3 * q0
    q1_q3 = 2 * q1 * q3
    q2_q0 = 2 * q2 * q0
    q2_q3 = 2 * q2 * q3
    q1_q0 = 2 * q1 * q0
    R = np.zeros((quaternion.shape[0], 3, 3))  # 创建一个n维数组来存储旋转矩阵
    R[:, 0, 0] = 1 - sq_q2 - sq_q3
    R[:, 0, 1] = q1_q2 - q3_q0
    R[:, 0, 2] = q1_q3 + q2_q0
    R[:, 1, 0] = q1_q2 + q3_q0
    R[:, 1, 1] = 1 - sq_q1 - sq_q3
    R[:, 1, 2] = q2_q3 - q1_q0
    R[:, 2, 0] = q1_q3 - q2_q0
    R[:, 2, 1] = q2_q3 + q1_q0
    R[:, 2, 2] = 1 - sq_q1 - sq_q2
    return R


def imu_data_to_world_coords(data, R):
    # 获取 data 矩阵的行数
    n = data.shape[0]
    # 初始化结果矩阵
    result = np.zeros((n, 3))
    # 遍历每一行的 data 和 R 进行点乘
    for i in range(n):
        result[i] = np.matmul(data[i], R[i])
    return result


# # 测试 [1,0,0,0]四元数
# rotation_and_show([0.78648,0.617316,-0.018206,-0.006167])
# # # 得到的旋转矩阵就代表 由标准状态到当前状态的变换
# a = [[0.78648,0.617316,-0.018206,-0.006167]]
# a = np.array(a)
# R = get_rotation_matrix_from_quaternion(a)
# print(R)
