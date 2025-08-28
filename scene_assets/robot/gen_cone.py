import math
import os


def generate_truncated_cone_with_material_y_up(
    obj_filename,
    bottom_radius=1.0,
    top_radius=0.6,
    height=0.5,
    segments=32,
    color=(0.5, 0.5, 0.5)
):  # 默认灰色
    """
    生成一个Y轴竖直向上的圆台的OBJ文件和MTL材质文件

    参数:
        obj_filename (str): 输出的OBJ文件名
        bottom_radius (float): 底面半径
        top_radius (float): 顶面半径
        height (float): 圆台高度
        segments (int): 圆周分段数
        color (tuple): RGB颜色值，范围0.0-1.0
    """
    vertices = []
    normals = []
    faces = []

    # 生成底面和顶面的顶点（注意：现在是在XZ平面上，Y轴向上）
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        # 底面顶点 (x, 0, z)
        vertices.append(
            (bottom_radius * cos_angle, 0.0, bottom_radius * sin_angle)
        )

        # 顶面顶点 (x, height, z)
        vertices.append(
            (top_radius * cos_angle, height, top_radius * sin_angle)
        )

        # 计算侧面法线
        normal_x = cos_angle
        normal_z = sin_angle
        # 考虑斜面角度的法线计算
        normal_y = (bottom_radius - top_radius) / height
        length = math.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        normals.append(
            (normal_x / length, normal_y / length, normal_z / length)
        )

    # 添加底面中心点和顶面中心点
    bottom_center_index = len(vertices) + 1
    vertices.append((0.0, 0.0, 0.0))  # 底面中心点

    top_center_index = len(vertices) + 1
    vertices.append((0.0, height, 0.0))  # 顶面中心点

    # 生成侧面的三角形
    for i in range(segments):
        v1 = i * 2 + 1
        v2 = i * 2 + 2
        v3 = (i * 2 + 3) % (segments * 2) + 1
        v4 = (i * 2 + 4) % (segments * 2) + 1

        # 每个侧面由两个三角形组成
        faces.append((v1, v3, v4))
        faces.append((v1, v4, v2))

        # 底面三角形
        faces.append((bottom_center_index, v1, v3))

        # 顶面三角形
        faces.append((top_center_index, v4, v2))

    # 创建MTL文件
    mtl_filename = os.path.splitext(obj_filename)[0] + ".mtl"
    with open(mtl_filename, 'w') as mtl_file:
        mtl_file.write("# 圆台材质文件\n\n")
        mtl_file.write("newmtl GrayMaterial\n")
        mtl_file.write(
            f"Ka {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
        )  # 环境光
        mtl_file.write(
            f"Kd {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
        )  # 漫反射
        mtl_file.write(f"Ks 0.500000 0.500000 0.500000\n")  # 镜面反射
        mtl_file.write("Ns 50.000000\n")  # 镜面高光系数
        mtl_file.write("d 1.000000\n")  # 不透明度
        mtl_file.write("illum 2\n")  # 光照模型

    # 写入OBJ文件
    with open(obj_filename, 'w') as obj_file:
        obj_file.write("# 圆台OBJ模型 - 杆状物体底座 (Y轴向上)\n")
        obj_file.write(f"# 底面半径: {bottom_radius}\n")
        obj_file.write(f"# 顶面半径: {top_radius}\n")
        obj_file.write(f"# 高度: {height}\n")
        obj_file.write(f"# 分段数: {segments}\n\n")

        # 引用MTL文件
        obj_file.write(f"mtllib {os.path.basename(mtl_filename)}\n\n")

        # 写入顶点
        for v in vertices:
            obj_file.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # 写入底面中心点和顶面中心点
        obj_file.write(f"v 0.000000 0.000000 0.000000\n")  # 底面中心点
        obj_file.write(f"v 0.000000 {height:.6f} 0.000000\n")  # 顶面中心点

        # 写入法线
        obj_file.write("vn 0.000000 -1.000000 0.000000\n")  # 底面法线 (Y轴向下)
        obj_file.write("vn 0.000000 1.000000 0.000000\n")  # 顶面法线 (Y轴向上)

        # 写入侧面法线
        for n in normals:
            obj_file.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        # 设置材质
        obj_file.write("\nusemtl GrayMaterial\n")

        # 写入面
        for face in faces:
            obj_file.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"Y轴向上的圆台OBJ文件已生成: {obj_filename}")
    print(f"材质MTL文件已生成: {mtl_filename}")
    print(f"底面半径: {bottom_radius}")
    print(f"顶面半径: {top_radius}")
    print(f"高度: {height}")
    print(f"分段数: {segments}")
    print(f"颜色: RGB({color[0]}, {color[1]}, {color[2]})")


# 使用示例
if __name__ == "__main__":
    generate_truncated_cone_with_material_y_up(
        "/data/zyw/workshop/attempt/scene_assets/robot/truncated_cone_base.obj",
        bottom_radius=0.5,  # 底面半径
        top_radius=0.3,  # 顶面半径
        height=0.5,  # 高度
        segments=256,  # 分段数，越高越平滑
        color=(0., 0., 0.)  # 灰色 (RGB值范围0-1)
    )
