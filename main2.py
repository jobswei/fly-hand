import numpy as np
import mujoco
import time
import mujoco.viewer

class FlyingHandForceController:
    def __init__(self, model, data, qpos_addr, max_force=10.0):
        """
        初始化飞行夹爪力控制器（无位置/速度硬限制）
        
        参数:
        - model: mujoco模型
        - data: mujoco数据
        - qpos_addr: 平动关节在qpos中的起始索引
        - max_force: 最大输出力（与xml中ctrlrange一致）
        """
        self.model = model
        self.data = data
        self.qpos_addr = qpos_addr  # 对应flying_hand_free关节的平动起始索引
        self.max_force = max_force  # 匹配xml的ctrlrange（如-10~10）
        
        # PID参数（针对力控制优化，z轴增强抗重力）
        self.pos_kp = np.array([80.0, 80.0, 100.0])  # 位置环比例增益
        self.pos_ki = np.array([0.5, 0.5, 1.0])      # 位置环积分增益（消除稳态误差）
        self.pos_kd = np.array([20.0, 20.0, 25.0])   # 位置环微分增益（抑制震荡）
        self.vel_kp = np.array([15.0, 15.0, 20.0])   # 速度环比例增益（快速响应速度偏差）
        self.vel_ki = np.array([1.0, 1.0, 1.5])      # 速度环积分增益
        self.vel_kd = np.array([5.0, 5.0, 8.0])      # 速度环微分增益
        
        # 控制状态变量
        self.target_pos = None       # 最终目标位置 [x, y, z]
        self.target_vel = None       # 最终目标速度 [vx, vy, vz]
        self.total_time = 0.0        # 从当前位置到目标的总运动时间
        self.current_time = 0.0      # 当前运动耗时
        self.start_pos = None        # 运动起始位置（首次step时记录）
        self.start_vel = None        # 运动起始速度（首次step时记录）
        self.moving = False          # 是否处于运动状态
        
        # PID中间变量（避免累积误差溢出）
        self.pos_integral = np.zeros(3)
        self.vel_integral = np.zeros(3)
        self.prev_pos_error = np.zeros(3)
        self.prev_vel_error = np.zeros(3)

    def set_target(self, target_pos, target_vel, total_time=3.0):
        """
        设置目标位置、目标速度和运动总时间
        
        参数:
        - target_pos: 期望到达的最终位置（无限制，完全遵循输入）
        - target_vel: 期望到达目标位置时的最终速度（无限制）
        - total_time: 完成从当前状态到目标状态的总时间（秒）
        """
        self.target_pos = np.array(target_pos, dtype=np.float64)
        self.target_vel = np.array(target_vel, dtype=np.float64)
        self.total_time = total_time
        self.current_time = 0.0
        self.moving = True
        
        # 重置PID状态（避免上一次运动的误差累积）
        self.pos_integral = np.zeros(3)
        self.vel_integral = np.zeros(3)
        self.prev_pos_error = np.zeros(3)
        self.prev_vel_error = np.zeros(3)
        self.start_pos = None  # 延迟到首次step时记录（确保获取实时起始位置）
        self.start_vel = None

    def get_current_state(self):
        """获取当前实时位置和速度（对应flying_hand_free关节的平动部分）"""
        # qpos: 自由关节通常存储为 [x, y, z, qw, qx, qy, qz]，取前3个为平动位置
        current_pos = self.data.qpos[self.qpos_addr:self.qpos_addr+3].copy()
        # qvel: 自由关节速度对应 [vx, vy, vz, wx, wy, wz]，取前3个为平动速度
        current_vel = self.data.qvel[self.qpos_addr:self.qpos_addr+3].copy()
        return current_pos, current_vel

    def compute_smooth_trajectory(self):
        """
        计算平滑的位置/速度轨迹（S型曲线）
        确保运动过程：从起始状态→平稳加速→过渡→平稳减速→到达目标状态
        """
        # 首次调用时记录起始状态（避免初始化时的位置偏差）
        if self.start_pos is None or self.start_vel is None:
            self.start_pos, self.start_vel = self.get_current_state()
        
        # 时间进度（0→1，超过1时按1处理，确保轨迹收尾）
        time_ratio = min(self.current_time / self.total_time, 1.0)
        
        # S型轨迹因子（三次多项式，确保速度连续：start→0，mid→max，end→0）
        # 位置因子：s(t) = 3t² - 2t³（t∈[0,1]，从0平滑过渡到1）
        pos_factor = 3 * (time_ratio ** 2) - 2 * (time_ratio ** 3)
        # 速度因子：ds/dt = 6t - 6t²（t∈[0,1]，从0→1→0，确保启停无冲击）
        vel_factor = 6 * time_ratio - 6 * (time_ratio ** 2)
        
        # 计算当前时刻的期望位置（从start_pos平滑过渡到target_pos）
        desired_pos = self.start_pos + (self.target_pos - self.start_pos) * pos_factor
        
        # 计算当前时刻的期望速度（从start_vel平滑过渡到target_vel）
        # 速度轨迹 = 起始速度 + 速度变化量×速度因子 + 位置偏差补偿（避免轨迹偏移）
        vel_change = self.target_vel - self.start_vel
        desired_vel = self.start_vel + vel_change * pos_factor + (self.target_pos - self.start_pos) * vel_factor / self.total_time
        
        return desired_pos, desired_vel

    def pid_compute_force(self):
        """双环PID计算控制力：位置环修正速度，速度环输出力"""
        current_pos, current_vel = self.get_current_state()
        desired_pos, desired_vel = self.compute_smooth_trajectory()
        dt = self.model.opt.timestep  # 仿真步长（从模型获取，确保时间精度）
        
        # -------------------------- 位置环（输出：速度修正量）--------------------------
        # 位置误差 = 期望位置 - 当前位置
        pos_error = desired_pos - current_pos
        # 积分项（累积位置误差，消除稳态偏差，限幅避免溢出）
        self.pos_integral += pos_error * dt
        self.pos_integral = np.clip(self.pos_integral, -1.0, 1.0)  # 软限幅防止积分饱和
        # 微分项（反映位置误差变化率，抑制震荡）
        pos_deriv = (pos_error - self.prev_pos_error) / dt
        # 位置环输出：速度修正量（让速度向“消除位置误差”的方向调整）
        vel_correction = self.pos_kp * pos_error + self.pos_ki * self.pos_integral + self.pos_kd * pos_deriv
        # 保存当前误差，用于下一次计算微分项
        self.prev_pos_error = pos_error.copy()
        
        # -------------------------- 速度环（输出：控制力）--------------------------
        # 最终速度指令 = 期望速度 + 位置环的速度修正量（兼顾轨迹和位置精度）
        final_vel_cmd = desired_vel + vel_correction
        # 速度误差 = 最终速度指令 - 当前速度
        vel_error = final_vel_cmd - current_vel
        # 积分项（累积速度误差，消除速度稳态偏差）
        self.vel_integral += vel_error * dt
        self.vel_integral = np.clip(self.vel_integral, -2.0, 2.0)  # 软限幅防止积分饱和
        # 微分项（反映速度误差变化率，抑制速度震荡）
        vel_deriv = (vel_error - self.prev_vel_error) / dt
        # 速度环输出：控制力（直接作用于执行器）
        control_force = self.vel_kp * vel_error + self.vel_ki * self.vel_integral + self.vel_kd * vel_deriv
        # 保存当前误差，用于下一次计算微分项
        self.prev_vel_error = vel_error.copy()
        
        # 限制控制力在执行器量程内（匹配xml的ctrlrange，避免超量程报错）
        return np.clip(control_force, -self.max_force, self.max_force)

    def step(self):
        """单步控制：计算力→应用到执行器→判断是否到达目标"""
        if not self.moving:
            return False  # 未处于运动状态，直接返回
        
        # 更新当前运动时间（累加仿真步长，确保时间进度准确）
        self.current_time += self.model.opt.timestep
        
        # 计算控制力并应用到执行器（对应fly_fx、fly_fy、fly_fz三个力执行器）
        control_force = self.pid_compute_force()
        self.data.ctrl[:3] = control_force  # 执行器顺序与xml一致：fx→fy→fz
        
        # -------------------------- 目标判断（时间结束时检查）--------------------------
        if self.current_time >= self.total_time:
            current_pos, current_vel = self.get_current_state()
            # 位置误差：当前位置与目标位置的欧氏距离
            pos_error = np.linalg.norm(current_pos - self.target_pos)
            # 速度误差：当前速度与目标速度的欧氏距离
            vel_error = np.linalg.norm(current_vel - self.target_vel)
            
            # 判定达标：位置误差<1cm（0.01m）且速度误差<2cm/s（0.02m/s）
            if pos_error < 0.01 and vel_error < 0.02:
                print(f"✅ 成功到达目标！")
                print(f"   位置误差：{pos_error:.4f}m | 速度误差：{vel_error:.4f}m/s")
            else:
                print(f"⚠️  时间结束，未完全达标！")
                print(f"   位置误差：{pos_error:.4f}m | 速度误差：{vel_error:.4f}m/s")
            
            self.moving = False  # 结束运动状态
            return True  # 标记本次运动完成
        
        return False  # 运动未结束


if __name__ == "__main__":
    # 1. 加载模型（替换为你的scene.xml路径）
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    
    # 2. 获取flying_hand_free关节的qpos起始索引（避免硬编码，提高兼容性）
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "flying_hand_free")
    qpos_addr = model.jnt_qposadr[joint_id]  # 自由关节的qpos起始地址
    
    # 3. 初始化控制器（max_force匹配xml中ctrlrange：如fly_fx/fy为-10~10，fly_fz为-10~100）
    controller = FlyingHandForceController(
        model=model,
        data=data,
        qpos_addr=qpos_addr,
        max_force=10.0  # 注意：若fly_fz需要更大力（如100），可单独调整z轴力限制（见下方注释）
    )
    # （可选）若需要单独调整z轴最大力（匹配xml的fly_fz=100）：
    # controller.max_force_z = 100.0
    # 在pid_compute_force()的control_force计算后添加：
    # control_force[2] = np.clip(control_force[2], -controller.max_force_z, controller.max_force_z)
    
    # 4. 设置目标（无位置/速度限制，完全按输入执行）
    # 目标位置：[x, y, z]，目标速度：[vx, vy, vz]，总运动时间：3秒
    controller.set_target(
        target_pos=[1.0, 1.0, 1.0],  # 你需要的目标位置
        target_vel=[0.0, 0.0, 0.0],  # 到达目标时的速度（此处设为静止）
        total_time=3.0               # 从当前位置到目标位置的总时间
    )
    
    # 5. 启动可视化仿真
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("📌 仿真启动，开始运动...")
        while viewer.is_running():
            # 执行仿真步（每次step推进一个仿真帧）
            mujoco.mj_step(model, data)
            # 执行控制器单步计算（更新力控制指令）
            motion_done = controller.step()
            # 同步可视化（确保视图与仿真数据一致）
            viewer.sync()
            # 控制仿真速度（避免过快，0.01秒/帧≈100FPS，可根据需要调整）
            time.sleep(0.01)
            
            # （可选）运动完成后可添加新目标
            if motion_done:
                print("\n📌 输入新目标（格式：x y z vx vy vz time），或输入q退出：")
                user_input = input("> ").strip()
                if user_input.lower() == "q":
                    break
                try:
                    # 解析用户输入的新目标（示例：0.5 0.5 1.2 0 0 0 2.5 → 位置(0.5,0.5,1.2)，速度(0,0,0)，时间2.5秒）
                    new_target = list(map(float, user_input.split()))
                    if len(new_target) == 7:
                        controller.set_target(
                            target_pos=new_target[:3],
                            target_vel=new_target[3:6],
                            total_time=new_target[6]
                        )
                    else:
                        print("❌ 输入格式错误！请输入7个数值：x y z vx vy vz time")
                except:
                    print("❌ 输入无效！请重新输入或输入q退出")