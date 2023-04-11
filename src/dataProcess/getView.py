import os
import numpy as np
import struct
import open3d
import time

from open3d import visualization


def save_view_point(pcd, filename):
    vis = visualization.Visualizer()
    vis.create_window(window_name='pcd',width=1440, height=1080)
    vis.add_geometry(pcd)

    vis.get_render_option().load_from_json('renderoption.json')
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    open3d.io.write_pinhole_camera_parameters(filename, param)
    # vis.destroy_window()


def load_view_point(pcd, filename):
    vis = visualization.Visualizer()
    vis.create_window(window_name='pcd',width=1440, height=1080)
    ctr = vis.get_view_control()
    param = open3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json('renderoption.json')
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    pcd =open3d.geometry.TriangleMesh.create_coordinate_frame(
    size=20, origin=[-2, -2, -2])
    save_view_point(pcd, "D:\DataSet\RIDI\Projetct\s\draw.json")  # 保存好得json文件位置
    load_view_point(pcd, "D:\DataSet\RIDI\Projetct\source\draw.json")  # 加载修改时较后的pcd文件

