# import numpy as np
# from scipy.spatial.transform import Rotation as R  # 用于验证（可选）

# mj_quat = (0.5, 0.5, -0.5, 0.5)
# scipy_quat = np.array([mj_quat[1], mj_quat[2], mj_quat[3], mj_quat[0]])
# rot = R.from_quat(scipy_quat)
# original_vector = np.array([0.041697, 2.4368E-05, 0.00014464])
# scipy_rotated = rot.apply(original_vector)
# print(scipy_rotated)

import numpy as np
import trimesh
from trimesh.transformations import quaternion_matrix, rotation_matrix


def rotate_stl_two_steps(input_stl_path, output_stl_path):
    """
    对STL执行两次连续旋转：
    第一步：绕(1,1,-1)轴旋转120° → 第二步：绕全局Y轴旋转180°
    """
    # 1. 读取原始STL
    mesh = trimesh.load(input_stl_path, file_type="stl")
    print(f"读取STL：{input_stl_path}，顶点数：{len(mesh.vertices)}")

    # ------------------- 第一步：绕(1,1,-1)轴旋转120° -------------------
    # 目标四元数（MuJoCo格式：w=0.5, x=0.5, y=-0.5, z=0.5）
    quat_step1 = np.array([0.5, 0.5, -0.5, 0.5])
    # 转换为trimesh格式（x,y,z,w）并生成旋转矩阵（4x4 → 取前3x3）
    trimesh_quat1 = [
        quat_step1[1], quat_step1[2], quat_step1[3], quat_step1[0]
    ]
    R1 = quaternion_matrix(trimesh_quat1)[:3, :3]  # 3x3旋转矩阵

    # ------------------- 第二步：绕全局Y轴旋转180° -------------------
    # 绕Y轴旋转180°的旋转矩阵（trimesh的rotation_matrix：轴=1代表Y轴，角度=π弧度=180°）
    R2 = rotation_matrix(angle=np.pi, direction=[0, 1, 0])[:3, :3]  # 3x3旋转矩阵

    # ------------------- 计算总旋转矩阵（先step1再step2） -------------------
    R_total = np.dot(R2, R1)  # 总旋转 = R2 × R1（顺序不能颠倒）

    # ------------------- 对所有顶点执行总旋转 -------------------
    rotated_vertices = np.dot(mesh.vertices,
                              R_total.T)  # (n,3) × (3,3) → (n,3)

    # ------------------- 导出最终STL -------------------
    rotated_mesh = trimesh.Trimesh(
        vertices=rotated_vertices,
        faces=mesh.faces,  # 面片连接关系不变
        metadata=mesh.metadata)
    rotated_mesh.export(output_stl_path, file_type="stl")
    print(f"最终STL已导出：{output_stl_path}")


# ------------------- 调用函数：旋转你的STL文件 -------------------
if __name__ == "__main__":
    # 目标四元数（MuJoCo格式：w=0.5, x=0.5, y=-0.5, z=0.5）
    target_quat = np.array([0.5, 0.5, -0.5, 0.5])

    # 批量处理多个STL（示例：处理link1、link2、link3）
    stl_files = [
        ("scene_assets/grippers/arx_x5/meshes/link1.stl",
         "scene_assets/grippers/arx_x5/meshes/link1_rotated.stl"
         ),  # 输入路径 → 输出路径
    ]

    # 逐个旋转并导出
    for input_path, output_path in stl_files:
        rotate_stl_two_steps(input_path, output_path)
