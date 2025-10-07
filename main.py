from blenderrenderer import Renderer
#!/usr/bin/env python3
import pickle as pkl
import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
from scipy.spatial.transform import Rotation

import yaml


def read_yaml(file_path):
    """
    读取 YAML 文件并返回 Python 数据结构
    :param file_path: YAML 文件的路径（相对路径或绝对路径）
    :return: 解析后的 Python 字典/列表
    """
    try:
        # 以只读模式打开 YAML 文件（指定 encoding='utf-8' 避免中文乱码）
        with open(file_path, 'r', encoding='utf-8') as f:
            # 解析 YAML 内容，返回 Python 数据
            yaml_data = yaml.safe_load(f)  # 推荐用 safe_load，避免执行恶意代码
        return yaml_data
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"错误：YAML 文件格式无效 - {e}")
        return None


class FlyingHandController:

    def __init__(
        self,
        xml_path='robotiq_2f85/fly_hand_scene.xml',
        render: Renderer = None,
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Find freejoint qpos indices
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                     'flying_hand_free')
        self.qpos_addr = self.model.jnt_qposadr[joint_id]

        # Initial pose
        self.target_pos = np.array([0.0, 0.0, 0.5])  # x, y, z
        self.target_euler = np.array([0.0, 0.0,
                                      0.0])  # roll, pitch, yaw in radians
        self.gripper_value = 0.0  # 0: open, 255: closed

        self.running = True

        # Trajectory tracking
        self.trajectory_mode = False
        self.trajectory = None
        self.render = render
        self.step = 0

    def set_pose(self,
                 x=None,
                 y=None,
                 z=None,
                 roll=None,
                 pitch=None,
                 yaw=None,
                 direction=None):
        """Set target pose. Angles in degrees. Direction is a 3D vector."""
        if x is not None: self.target_pos[0] = x
        if y is not None: self.target_pos[1] = y
        if z is not None: self.target_pos[2] = z

        # Update position
        self.data.qpos[self.qpos_addr:self.qpos_addr + 3] = self.target_pos

        # Handle rotation
        if direction is not None:
            # Convert direction vector to quaternion
            # Assuming direction is the forward vector (z-axis of gripper)
            direction = np.array(direction)
            direction = direction / np.linalg.norm(direction)  # Normalize

            # Default up vector (world y-axis)
            up = np.array([0, 0, 1])

            # If direction is parallel to up, use different up vector
            if abs(np.dot(direction, up)) > 0.99:
                up = np.array([1, 0, 0])

            # Compute right vector
            right = np.cross(up, direction)
            right = right / np.linalg.norm(right)

            # Recompute up vector
            up = np.cross(direction, right)

            # Build rotation matrix (columns are right, up, forward)
            rot_matrix = np.column_stack([right, up, direction])
            rot = Rotation.from_matrix(rot_matrix)
            quat_xyzw = rot.as_quat()
            quat_wxyz = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            self.data.qpos[self.qpos_addr + 3:self.qpos_addr + 7] = quat_wxyz
        else:
            # Use euler angles if no direction vector provided
            if roll is not None: self.target_euler[0] = np.deg2rad(roll)
            if pitch is not None: self.target_euler[1] = np.deg2rad(pitch)
            if yaw is not None: self.target_euler[2] = np.deg2rad(yaw)

            # Convert euler to quaternion
            rot = Rotation.from_euler('xyz', self.target_euler)
            quat_xyzw = rot.as_quat()
            quat_wxyz = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            self.data.qpos[self.qpos_addr + 3:self.qpos_addr + 7] = quat_wxyz

    def set_gripper(self, value):
        """Set gripper opening. 0=open, 255=closed"""
        self.gripper_value = np.clip(value, 0, 255)
        if hasattr(self.data, 'ctrl') and len(self.data.ctrl) > 6:
            self.data.ctrl[
                6] = self.gripper_value  # Assuming gripper is the 7th actuator

    def load_trajectory(self, filename):
        with open(filename, 'rb') as f:
            self.trajectory = pkl.load(f)
        print(f"Trajectory loaded from {filename}")
        return self.trajectory

    def start_trajectory(self, traj_type='circle', duration=10.0):
        """Start trajectory tracking mode"""
        self.trajectory = self.load_trajectory(traj_type)
        if self.trajectory:
            self.trajectory_mode = True
            self.step = 0
            print(f"Started {traj_type} trajectory for {duration} seconds")

    def stop_trajectory(self):
        """Stop trajectory tracking"""
        self.trajectory_mode = False
        self.trajectory = None
        self.step = 0
        print("Trajectory stopped")

    def update_trajectory(self):
        """Update position based on current trajectory"""
        if not self.trajectory_mode or not self.trajectory:
            return

        self.step += 1
        t = self.step % len(self.trajectory['positions'])

        # Set position and direction vector
        self.set_pose(x=self.trajectory['positions'][t, 0],
                      y=self.trajectory['positions'][t, 1],
                      z=self.trajectory['positions'][t, 2],
                      direction=self.trajectory['directions'][t])

    def run(self):
        """Main loop with viewer"""
        # Start input thread
        # Set initial pose
        self.set_pose()
        self.start_trajectory('traj.pkl')
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and self.running:
                # Update trajectory if active
                self.update_trajectory()

                # Step simulation
                mujoco.mj_step(self.model, self.data)
                if self.render is not None and self.step % 10 == 0:
                    positions = self.data.xpos[1:]
                    quats = self.data.xquat[1:]
                    rotations = []
                    for quat in quats:
                        quat_xyzw = np.array(
                            [quat[1], quat[2], quat[3], quat[0]])
                        rot = Rotation.from_quat(quat_xyzw)
                        euler_angles = rot.as_euler('xyz')
                        euler_degrees = np.rad2deg(euler_angles)
                        rotations.append(euler_degrees)
                    print("render....")
                    self.render.render(positions, rotations,
                                       f"work_dir/test_{self.step}.png")
                    print("render done")
                # Update viewer
                viewer.sync()

                # Small delay to control simulation speed
                time.sleep(0.01)

        print("/nExiting...")


object_config = {
    "objects": [{
        "asset":
        "coffee_cup",
        "position":
        [0.6175566804667381, 0.4018943018251386, 0.31937398627448127],
        "rotation": [90, 0, 0],
        "size":
        0.098
    }, {
        "asset":
        "yellow_barrier",
        "position":
        [0.05130502865257103, 0.6606247730319297, 0.31637398627448127],
        "rotation": [90, 0, 0],
        "size":
        0.092
    }, {
        "asset":
        "book",
        "position":
        [0.06635277244501003, 0.473959091311956, 0.31437398627448127],
        "rotation": [90, 0, 0],
        "size":
        0.088
    }, {
        "asset":
        "football",
        "position":
        [-0.3885905482642662, 0.5489812188031324, 0.4033183338783049],
        "rotation": [90, 0, 0],
        "size":
        0.26588869520764724
    }, {
        "asset":
        "football",
        "position":
        [-0.3900378037115135, 0.8267650265556323, 0.4148765651537514],
        "rotation": [90, 0, 0],
        "size":
        0.2890051577585402
    }, {
        "asset":
        "basketball",
        "position":
        [-0.18117858987528535, 0.19665202236078264, 0.35448618047047975],
        "rotation": [90, 0, 0],
        "size":
        0.168224388391997
    }, {
        "asset": "table",
        "position": [0, 0.75, -0.06662601372551874],
        "rotation": [90, 0, 0],
        "size": 2
    }]
}
asset_paths = {
    'coffee_cup':
    "./scene_assets/living_room/Coffee_cup_withe_obj/Coffee_cup_withe_.obj",
    'yellow_barrier':
    "./scene_assets/living_room/Cone_Buoy/conbyfr.obj",
    'book':
    "./scene_assets/living_room/Book_by_Peter_Iliev_obj/Book_by_Peter_Iliev_obj.obj",
    'football':
    "./scene_assets/living_room/soccer/Obj.obj",
    'basketball':
    "./scene_assets/living_room/basketball/BasketBall.obj",
    'table':
    "scene_assets/env_assets/CENSI_COFTABLE_obj/Censi_ConcreteCoffeeTable_free_obj.obj",
    "2f85_base_mount":
    "scene_assets/grippers/robotiq_2f85/assets/base_mount.stl",
    "2f85_base":
    "scene_assets/grippers/robotiq_2f85/assets/base.stl",
    "2f85_driver":
    "scene_assets/grippers/robotiq_2f85/assets/driver.stl",
    "2f85_coupler":
    "scene_assets/grippers/robotiq_2f85/assets/coupler.stl",
    "2f85_spring_link":
    "scene_assets/grippers/robotiq_2f85/assets/spring_link.stl",
    "2f85_follower":
    "scene_assets/grippers/robotiq_2f85/assets/follower.stl",
    "2f85_pad":
    "scene_assets/grippers/robotiq_2f85/assets/pad.stl",
    "2f85_silicone_pad":
    "scene_assets/grippers/robotiq_2f85/assets/silicone_pad.stl"
}
render = Renderer(
    [0, 4, 3],
    60, [0, 0, 0],
    has_wall=False,
    plane_texture=
    r"D:/files/codes/fly-hand/scene_assets/textures/rustic_floor.png",
    wall_texture=r"D:/files/codes/fly-hand/scene_assets/textures/gray_wall.png"
)

# positions = [obj["position"] for obj in object_config["objects"]]
# rotations = [obj["rotation"] for obj in object_config["objects"]]
# render.render(positions, rotations, "test.png")

griper_config = read_yaml("scene_assets/grippers/robotiq_2f85/config.yaml")
positions = []
rotations = []
for config in griper_config:
    render._setup_object(asset_path=asset_paths[config["asset"]],
                         size=config["size"],
                         color=config["color"])
    positions.append(config["position"])
    rot = Rotation.from_quat(config["rotation"])
    euler_angles = rot.as_euler('xyz')
    euler_degrees = np.rad2deg(euler_angles)
    rotations.append(euler_degrees)
for obj in object_config["objects"]:
    render._setup_object(
        asset_path=asset_paths[obj["asset"]],
        size=obj["size"],
    )
controller = FlyingHandController('scene.xml', None)
controller.run()
# render.render(positions, rotations, "test.png")
