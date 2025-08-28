#!/usr/bin/env python3
import pickle as pkl
import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
from scipy.spatial.transform import Rotation

class FlyingHandController:
    def __init__(self, xml_path='robotiq_2f85/fly_hand_scene.xml'):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Find freejoint qpos indices
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'flying_hand_free')
        self.qpos_addr = self.model.jnt_qposadr[joint_id]
        
        # Initial pose
        self.target_pos = np.array([0.0, 0.0, 0.5])  # x, y, z
        self.target_euler = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw in radians
        self.gripper_value = 0.0  # 0: open, 255: closed
        
        self.running = True
        
        # Trajectory tracking
        self.trajectory_mode = False
        self.trajectory = None
        
    def set_pose(self, x=None, y=None, z=None, roll=None, pitch=None, yaw=None, direction=None):
        """Set target pose. Angles in degrees. Direction is a 3D vector."""
        if x is not None: self.target_pos[0] = x
        if y is not None: self.target_pos[1] = y
        if z is not None: self.target_pos[2] = z
        
        # Update position
        self.data.qpos[self.qpos_addr:self.qpos_addr+3] = self.target_pos
        
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
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            self.data.qpos[self.qpos_addr+3:self.qpos_addr+7] = quat_wxyz
        else:
            # Use euler angles if no direction vector provided
            if roll is not None: self.target_euler[0] = np.deg2rad(roll)
            if pitch is not None: self.target_euler[1] = np.deg2rad(pitch)
            if yaw is not None: self.target_euler[2] = np.deg2rad(yaw)
            
            # Convert euler to quaternion
            rot = Rotation.from_euler('xyz', self.target_euler)
            quat_xyzw = rot.as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            self.data.qpos[self.qpos_addr+3:self.qpos_addr+7] = quat_wxyz
        
    def set_gripper(self, value):
        """Set gripper opening. 0=open, 255=closed"""
        self.gripper_value = np.clip(value, 0, 255)
        if hasattr(self.data, 'ctrl') and len(self.data.ctrl) > 6:
            self.data.ctrl[6] = self.gripper_value  # Assuming gripper is the 7th actuator
    
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
                
                # Update viewer
                viewer.sync()
                
                # Small delay to control simulation speed
                time.sleep(0.01)
                
        print("\nExiting...")

if __name__ == "__main__":
    controller = FlyingHandController('scene.xml')
    controller.run()