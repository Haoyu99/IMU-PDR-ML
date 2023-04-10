import open3d as o3d

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=20, origin=[-2, -2, -2])
pcd = o3d.io.read_point_cloud("src/dataProcess/resources/iPhonex.pcd")
# th = np.array([0, 0, -math.pi/2]).T
# thAxis = pcd.get_rotation_matrix_from_axis_angle(th)
# pcd.rotate(thAxis)

o3d.visualization.draw_geometries([mesh_frame, pcd])
# o3d.io.write_point_cloud("iPhonex.pcd", pcd)
