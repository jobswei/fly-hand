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

Vector3D = tuple[float]


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


class PIDPositionController:

    def __init__(
            self,
            kp_pos,
            ki_pos,
            kd_pos,  # 位置环参数
            kp_vel,
            ki_vel,
            kd_vel,  # 速度环参数
            max_output,
            min_output):  # 输出限制

        # 位置环PID参数
        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kd_pos = kd_pos

        # 速度环PID参数
        self.kp_vel = kp_vel
        self.ki_vel = ki_vel
        self.kd_vel = kd_vel

        # 输出限制
        self.max_output = max_output
        self.min_output = min_output

        # 位置环变量
        self.pos_error = 0.0
        self.pos_integral = 0.0
        self.pos_derivative = 0.0
        self.prev_pos_error = 0.0

        # 速度环变量
        self.vel_error = 0.0
        self.vel_integral = 0.0
        self.vel_derivative = 0.0
        self.prev_vel_error = 0.0

    def compute(self, current_pos, target_pos, current_vel, target_vel, dt):
        """
        计算PID输出
        current_pos: 当前位置
        target_pos: 目标位置
        current_vel: 当前速度
        target_vel: 目标速度
        dt: 时间间隔
        """
        # 计算位置误差
        self.pos_error = target_pos - current_pos

        # 计算位置环积分项（带抗积分饱和）
        self.pos_integral += self.pos_error * dt
        # 限制积分项，防止饱和
        self.pos_integral = self._clamp(
            self.pos_integral,
            self.min_output / self.ki_pos if self.ki_pos != 0 else 0,
            self.max_output / self.ki_pos if self.ki_pos != 0 else 0)

        # 计算位置环微分项
        if dt > 0:
            self.pos_derivative = (self.pos_error - self.prev_pos_error) / dt
        else:
            self.pos_derivative = 0.0

        # 保存当前误差用于下次计算微分
        self.prev_pos_error = self.pos_error

        # 计算位置环输出作为速度环的目标
        vel_setpoint = (self.kp_pos * self.pos_error +
                        self.ki_pos * self.pos_integral +
                        self.kd_pos * self.pos_derivative)

        # 计算速度误差（使用位置环输出作为目标）
        self.vel_error = vel_setpoint - current_vel

        # 计算速度环积分项（带抗积分饱和）
        self.vel_integral += self.vel_error * dt
        self.vel_integral = self._clamp(
            self.vel_integral,
            self.min_output / self.ki_vel if self.ki_vel != 0 else 0,
            self.max_output / self.ki_vel if self.ki_vel != 0 else 0)

        # 计算速度环微分项
        if dt > 0:
            self.vel_derivative = (self.vel_error - self.prev_vel_error) / dt
        else:
            self.vel_derivative = 0.0

        # 保存当前误差用于下次计算微分
        self.prev_vel_error = self.vel_error

        # 计算速度环输出（最终控制量）
        output = (self.kp_vel * self.vel_error +
                  self.ki_vel * self.vel_integral +
                  self.kd_vel * self.vel_derivative)

        # 限制输出
        return self._clamp(output, self.min_output, self.max_output)

    def _clamp(self, value, min_val, max_val):
        """限制值在[min_val, max_val]范围内"""
        return max(min_val, min(value, max_val))


class PIDPositionController3D:

    def __init__(self, kp_pos: float, ki_pos: float, kd_pos: float,
                 kp_vel: float, ki_vel: float, kd_vel: float,
                 max_output: float, min_output: float):
        self.pid_x = PIDPositionController(kp_pos, ki_pos, kd_pos, kp_vel,
                                           ki_vel, kd_vel, max_output,
                                           min_output)
        self.pid_y = PIDPositionController(kp_pos, ki_pos, kd_pos, kp_vel,
                                           ki_vel, kd_vel, max_output,
                                           min_output)
        self.pid_z = PIDPositionController(kp_pos, ki_pos, kd_pos, kp_vel,
                                           ki_vel, kd_vel, max_output,
                                           min_output)

    def compute(self, current_pos: Vector3D, target_pos: Vector3D,
                current_vel: Vector3D, target_vel: Vector3D,
                dt: float) -> Vector3D:
        output_x = self.pid_x.compute(current_pos[0], target_pos[0],
                                      current_vel[0], target_vel[0], dt)
        output_y = self.pid_y.compute(current_pos[1], target_pos[1],
                                      current_vel[1], target_vel[1], dt)
        output_z = self.pid_z.compute(current_pos[2], target_pos[2],
                                      current_vel[2], target_vel[2], dt)
        return np.array([output_x, output_y, output_z])


def quaternion_multiply(q1, q2):
    """
    四元数乘法
    q1, q2: 四元数，格式为 [w, x, y, z]
    返回: 乘积 q1 * q2，格式为 [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def quaternion_conjugate(q):
    """计算四元数的共轭"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵"""
    w, x, y, z = q
    return np.array([[
        1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w
    ], [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
                     [
                         2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w,
                         1 - 2 * x**2 - 2 * y**2
                     ]])


def quaternion_error(q_current, q_target):
    """
    计算当前姿态到目标姿态的误差四元数
    返回: 误差四元数，表示从当前姿态到目标姿态的旋转
    """
    # 误差四元数 = 目标四元数 * 当前四元数的共轭
    q_error = quaternion_multiply(q_target, quaternion_conjugate(q_current))

    # 确保我们取最短路径（如果实部为负，取共轭）
    if q_error[0] < 0:
        q_error = -q_error

    return q_error


def quaternion_to_angular_velocity_error(q_error):
    """
    将四元数误差转换为角速度误差向量
    对于小角度，这近似于旋转轴乘以旋转角度
    """
    # 提取虚部作为旋转轴
    axis = q_error[1:]
    # 计算旋转角度
    angle = 2 * np.arccos(np.clip(q_error[0], -1.0, 1.0))

    # 对于小角度，使用近似；对于大角度，使用精确计算
    if angle < 1e-6:
        return 2 * axis  # 近似：sin(angle/2) ≈ angle/2
    else:
        return axis * angle / np.sin(angle / 2)


class QuaternionPIDController:

    def __init__(
            self,
            kp_angle,
            ki_angle,
            kd_angle,  # 角度环PID参数
            kp_rate,
            ki_rate,
            kd_rate,  # 角速度环PID参数
            max_torque,
            min_torque):  # 输出力矩限制

        # 角度环PID参数（三维，分别对应x, y, z轴）
        self.kp_angle = np.array(kp_angle) if isinstance(
            kp_angle, (list, np.ndarray)) else np.array([kp_angle] * 3)
        self.ki_angle = np.array(ki_angle) if isinstance(
            ki_angle, (list, np.ndarray)) else np.array([ki_angle] * 3)
        self.kd_angle = np.array(kd_angle) if isinstance(
            kd_angle, (list, np.ndarray)) else np.array([kd_angle] * 3)

        # 角速度环PID参数（三维）
        self.kp_rate = np.array(kp_rate) if isinstance(
            kp_rate, (list, np.ndarray)) else np.array([kp_rate] * 3)
        self.ki_rate = np.array(ki_rate) if isinstance(
            ki_rate, (list, np.ndarray)) else np.array([ki_rate] * 3)
        self.kd_rate = np.array(kd_rate) if isinstance(
            kd_rate, (list, np.ndarray)) else np.array([kd_rate] * 3)

        # 输出力矩限制
        self.max_torque = np.array(max_torque) if isinstance(
            max_torque, (list, np.ndarray)) else np.array([max_torque] * 3)
        self.min_torque = np.array(min_torque) if isinstance(
            min_torque, (list, np.ndarray)) else np.array([min_torque] * 3)

        # 角度环变量
        self.angle_error = np.zeros(3)
        self.angle_integral = np.zeros(3)
        self.angle_derivative = np.zeros(3)
        self.prev_angle_error = np.zeros(3)

        # 角速度环变量
        self.rate_error = np.zeros(3)
        self.rate_integral = np.zeros(3)
        self.rate_derivative = np.zeros(3)
        self.prev_rate_error = np.zeros(3)

    def compute(self, current_quat, target_quat, current_ang_vel,
                target_ang_vel, dt):
        """
        计算PID输出力矩
        current_quat: 当前姿态四元数 [w, x, y, z]
        target_quat: 目标姿态四元数 [w, x, y, z]
        current_ang_vel: 当前角速度 [wx, wy, wz]
        target_ang_vel: 目标角速度 [wx, wy, wz]
        dt: 时间间隔
        """
        # 标准化四元数
        current_quat = current_quat / np.linalg.norm(current_quat)
        target_quat = target_quat / np.linalg.norm(target_quat)

        # 计算姿态误差四元数
        q_err = quaternion_error(current_quat, target_quat)

        # 将姿态误差转换为角度误差向量
        self.angle_error = quaternion_to_angular_velocity_error(q_err)

        # 计算角度环积分项（带抗积分饱和）
        self.angle_integral += self.angle_error * dt
        for i in range(3):
            if self.ki_angle[i] != 0:
                self.angle_integral[i] = self._clamp(
                    self.angle_integral[i],
                    self.min_torque[i] / self.ki_angle[i],
                    self.max_torque[i] / self.ki_angle[i])

        # 计算角度环微分项
        if dt > 0:
            self.angle_derivative = (self.angle_error -
                                     self.prev_angle_error) / dt
        else:
            self.angle_derivative = np.zeros(3)

        # 保存当前误差用于下次计算微分
        self.prev_angle_error = self.angle_error.copy()

        # 计算角度环输出作为角速度环的目标
        rate_setpoint = (self.kp_angle * self.angle_error +
                         self.ki_angle * self.angle_integral + self.kd_angle *
                         self.angle_derivative) + target_ang_vel

        # 计算角速度误差
        self.rate_error = rate_setpoint - current_ang_vel

        # 计算角速度环积分项（带抗积分饱和）
        self.rate_integral += self.rate_error * dt
        for i in range(3):
            if self.ki_rate[i] != 0:
                self.rate_integral[i] = self._clamp(
                    self.rate_integral[i],
                    self.min_torque[i] / self.ki_rate[i],
                    self.max_torque[i] / self.ki_rate[i])

        # 计算角速度环微分项
        if dt > 0:
            self.rate_derivative = (self.rate_error -
                                    self.prev_rate_error) / dt
        else:
            self.rate_derivative = np.zeros(3)

        # 保存当前误差用于下次计算微分
        self.prev_rate_error = self.rate_error.copy()

        # 计算角速度环输出（最终力矩）
        torque = (self.kp_rate * self.rate_error +
                  self.ki_rate * self.rate_integral +
                  self.kd_rate * self.rate_derivative)

        # 限制输出力矩
        for i in range(3):
            torque[i] = self._clamp(torque[i], self.min_torque[i],
                                    self.max_torque[i])

        return torque

    def _clamp(self, value, min_val, max_val):
        """限制值在[min_val, max_val]范围内"""
        return max(min_val, min(value, max_val))


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
        self.qpos_addr = self.model.jnt_qposadr[joint_id]
        self.gripper_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
        self.fx_id = mujoco.mj_name2id(self.model,
                                       mujoco.mjtObj.mjOBJ_ACTUATOR, "fx")
        self.mx_id = mujoco.mj_name2id(self.model,
                                       mujoco.mjtObj.mjOBJ_ACTUATOR, "mx")
        self.running = True
        self.debug_step = 0

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

    def get_current_pose(self):
        self.step += 1
        t = self.step % len(self.trajectory['positions'])
        x = self.trajectory['positions'][t, 0]
        y = self.trajectory['positions'][t, 1]
        z = self.trajectory['positions'][t, 2]
        return (x, y, z)

    def run(self):
        """Main loop with viewer"""
        # Start input thread
        # Set initial pose
        self.set_pose()
        self.start_trajectory('traj.pkl')
        body_names = []
        for i in range(self.model.nbody):
            # 通过body_id获取对应的名称
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                          i)
            body_names.append(body_name)
        mass = sum(self.model.body_mass[1:15])
        pid = PIDPositionController3D(kp_pos=30,
                                      ki_pos=0.1,
                                      kd_pos=1.0,
                                      kp_vel=5.0,
                                      ki_vel=0.5,
                                      kd_vel=0.5,
                                      max_output=10.0,
                                      min_output=-10.0)
        pid_rot = QuaternionPIDController(
            kp_angle=[5.0, 5.0, 5.0],  # 角度环比例参数
            ki_angle=[0.1, 0.1, 0.1],  # 角度环积分参数
            kd_angle=[2.0, 2.0, 2.0],  # 角度环微分参数
            kp_rate=[2.0, 2.0, 2.0],  # 角速度环比例参数
            ki_rate=[0.05, 0.05, 0.05],  # 角速度环积分参数
            kd_rate=[0.5, 0.5, 0.5],  # 角速度环微分参数
            max_torque=1.0,  # 最大力矩限制
            min_torque=-1.0  # 最小力矩限制
        )
        self.update_trajectory()
        target_position = self.get_current_pose()
        # target_position = (0.3, 0.5, 0.3)
        target_rotation = (1, 0, 0, 0)  # No rotation (identity quaternion)
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and self.running:
                # Update trajectory if active
                if self.debug_step % 1 == 0:
                    target_position = self.get_current_pose()
                self.debug_step += 1
                self.data.ctrl[self.gripper_actuator_id] = 255
                # print(self.data.qpos[self.qpos_addr:self.qpos_addr + 3])
                output = pid.compute(
                    tuple(self.data.qpos[self.qpos_addr:self.qpos_addr + 3]),
                    target_position,
                    tuple(self.data.qvel[self.qpos_addr:self.qpos_addr + 3]),
                    (0, 0, 0), self.model.opt.timestep)
                gravity = -self.model.opt.gravity * mass
                self.data.ctrl[self.fx_id:self.fx_id +
                               3] = np.array(output) + gravity
                ############# For rotation control #############
                # output = pid_rot.compute(
                #     self.data.qpos[self.qpos_addr + 3:self.qpos_addr + 7],
                #     target_rotation,
                #     self.data.qvel[self.qpos_addr + 3:self.qpos_addr + 6],
                #     (0, 0, 0), self.model.opt.timestep)
                # self.data.ctrl[self.mx_id:self.mx_id + 3] = output
                # self.data.ctrl[self.fx_id] = 0.5
                # self.data.ctrl[self.vx_id] = 0.0
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                print(
                    np.linalg.norm(
                        self.data.qpos[self.qpos_addr:self.qpos_addr + 3] -
                        target_position) / np.linalg.norm(target_position))

                # self.update_trajectory()
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
# render = Renderer(
#     [0, 4, 3],
#     60, [0, 0, 0],
#     has_wall=False,
#     plane_texture=
#     r"D:/files/codes/fly-hand/scene_assets/textures/rustic_floor.png",
#     wall_texture=r"D:/files/codes/fly-hand/scene_assets/textures/gray_wall.png"
# )

# positions = [obj["position"] for obj in object_config["objects"]]
# rotations = [obj["rotation"] for obj in object_config["objects"]]
# render.render(positions, rotations, "test.png")

# griper_config = read_yaml("scene_assets/grippers/robotiq_2f85/config.yaml")
# positions = []
# rotations = []
# for config in griper_config:
#     render._setup_object(asset_path=asset_paths[config["asset"]],
#                          size=config["size"],
#                          color=config["color"])
#     positions.append(config["position"])
#     rot = Rotation.from_quat(config["rotation"])
#     euler_angles = rot.as_euler('xyz')
#     euler_degrees = np.rad2deg(euler_angles)
#     rotations.append(euler_degrees)
# for obj in object_config["objects"]:
#     render._setup_object(
#         asset_path=asset_paths[obj["asset"]],
#         size=obj["size"],
#     )
controller = FlyingHandController('scene.xml', None)
controller.run()
# render.render(positions, rotations, "test.png")
