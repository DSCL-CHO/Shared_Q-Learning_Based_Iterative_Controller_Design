##TORQUE CONTROL=J@FORCE+GRAVITY      # 9D TORQUE
##TORQUE CONTROL=J@FORCE+GRAVITY      # 9D TORQUE
##TORQUE CONTROL=J@FORCE+GRAVITY      # 9D TORQUE

import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  
import pybullet as p
import pybullet_data
import numpy as np
import time
from kp_then_kd.coord_to_xyz import coord_to_xyz

class MultiPandaEnv:
    def __init__(self, num_robots=2, gui=True, x_offsets=None,
                 kp_list=None, kd_list=None,
                 draw_traj=False, traj_color=(1, 0, 0), traj_life=0, traj_width=2):  # 📏
        self.gui = gui
        self.num_robots = num_robots
     
        self.kp_list = kp_list if kp_list else [150.0] * num_robots  # 📏
        self.kd_list = kd_list if kd_list else [20.0] * num_robots   # 📏

        self.ee_link_index = 11
        self.robots = []
        self.prev_errors = []
        self.errors = [[] for _ in range(num_robots)]  # 로봇별 오차 저장 리스트 🔎

        # 궤적 그리기 옵션
        self.draw_traj = draw_traj
        self.traj_color = traj_color
        self.traj_life = traj_life
        self.traj_width = traj_width
        # 궤적 시작점 기록
        self.traj_last_pos = [None] * num_robots
        self.traj_ids = [[] for _ in range(num_robots)]

        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # p.setRealTimeSimulation(0)
        # p.setTimeStep(0.001)
        ##camera
        p.resetDebugVisualizerCamera(
            cameraDistance=6.0,               # 카메라와 타깃 거리
            cameraYaw=-20,                     # 수평 회전 (좌우 회전)
            cameraPitch=-33,                 # 수직 회전 (위아래 각도)
            cameraTargetPosition=[2.5, 1.5, -1]    # 카메라가 바라보는 중심
        )
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("/home/nvidia/env/env.urdf", basePosition=[0-0.1,-0.1,0.3], useFixedBase=True)

        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)   

        if x_offsets is None:
            x_offsets = [i * 1.0 for i in range(num_robots)]

        for i in range(num_robots):
            base_pos = [x_offsets[i], 0, 0.4]
            robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=base_pos, useFixedBase=True)
            self.robots.append(robot_id)
            self.prev_errors.append(np.zeros(3))
            if self.draw_traj:
                self.traj_last_pos[i] = self.get_ee_position(i)

                # 팔 7축(기존 movable_joints 유지)
        self.movable_joints = [
            j for j in range(p.getNumJoints(self.robots[0]))
            if p.getJointInfo(self.robots[0], j)[2] == p.JOINT_REVOLUTE
        ][:7]
        self.prev_vel = [np.zeros(len(self.movable_joints)) for _ in range(num_robots)]
        ###########################
        for rid in self.robots:
            for j in range(p.getNumJoints(rid)):
                p.setJointMotorControl2(
                    rid, j,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=0
                )
    
    def move_all(self, target_list, steps=100, dt=0.001):
    # def move_all(self, target_list, target_list_vel, steps=100, dt=0.001):

        for _ in range(steps):
            for idx in range(self.num_robots):
                robot_id = self.robots[idx]
                
                # 현재 관절 상태 업데이트
                joint_Pos = [float(p.getJointState(robot_id, j)[0]) for j in self.movable_joints] # 77777
                joint_vel = [float(p.getJointState(robot_id, j)[1]) for j in self.movable_joints]
                # self.prev_vel[idx]= joint_vel
                
                joint_acc = (joint_vel -  self.prev_vel[idx])/dt
                self.prev_vel[idx] = np.array(joint_vel) 

                
                cur = self.get_ee_position(idx)
                target = np.array(target_list[idx])   #3d
                error = target - cur

                cur_vel = self.get_ee_velocity(idx)   # ← () 붙여서 호출해야 함
                d_error = - np.array(cur_vel)
                # print(cur_vel)
                
                # joint_states = [p.getJointState(robot_id, j) for j in self.movable_joints]

                q = [*joint_Pos,0.0,0.0]
                qdot_g=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] # [q1-q7, prismatic joint 1, 2]
                qddot_g=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

                # gravity_torques=p.calculateInverseDynamics(self.robot, q, qddot, qddot)
                gravity_torques=p.calculateInverseDynamics(robot_id, q, qdot_g, qddot_g)
                
                qdot=[*joint_vel,0.0,0.0] # [q1-q7, prismatic joint 1, 2]
                qddot=[*joint_acc,0.0,0.0] 

                jacobian=p.calculateJacobian(
                    # self.robots,
                    robot_id,
                    self.ee_link_index,
                    [0.0, 0.0, 0.0],   # local position: 길이 3 리스트
                    q,        
                    qdot,       
                    qddot          
                )
                # print(jacobian)
                
                
                # jacobian = (linear_jacobian, angular_jacobian)
                linear_jacobian, angular_jacobian = jacobian[0], jacobian[1]
                J = np.array(linear_jacobian) 

                force = self.kp_list[idx] * error + self.kd_list[idx] * d_error
                tau = J.T @ force  + gravity_torques 
                #9*1=9*3@3*1 + 9*1
                # print(gravity_torques)   ## 9D
                # print(tau)               ##9D
                # print("force=",force)       ##3D
                # print("J=",J)               
                

                for i, j in enumerate(self.movable_joints):
                    p.setJointMotorControl2(robot_id,  j, p.TORQUE_CONTROL, force=float(tau[i]))
                
                err_norm = np.linalg.norm(error)   #  🔎
                self.errors[idx].append(err_norm)  #  🔎

            p.stepSimulation()

            # 궤적 선 그리기
            if self.draw_traj:
                for idx in range(self.num_robots):
                    new_pos = self.get_ee_position(idx)
                    last = self.traj_last_pos[idx]
                    if last is not None:
                        line_id = p.addUserDebugLine(
                            last, new_pos,
                            lineColorRGB=self.traj_color,
                            lineWidth=self.traj_width,
                            lifeTime=self.traj_life
                        )
                        self.traj_ids[idx].append(line_id)
                    self.traj_last_pos[idx] = new_pos


            if self.gui:
                time.sleep(1/240)
    def reset_trajectory(self):
        """그려진 선 초기화 & 현재 위치를 시작점으로 설정"""
        for idx in range(self.num_robots):
            for lid in self.traj_ids[idx]:
                try:
                    p.removeUserDebugItem(lid)
                except Exception:
                    pass
            self.traj_ids[idx].clear()
            self.traj_last_pos[idx] = self.get_ee_position(idx)

    def disconnect(self):
        p.disconnect()

    def render_grid_overlay(self, x_offset=0.0, grid_shape=(4, 4), cell_size=0.1,
                            reward_map=None, wall_states=None, goal_state=None):
        margin = 0.005
        box_half_size = cell_size / 2 - margin
        for row in range(grid_shape[0]):
            for col in range(grid_shape[1]):
                pos = coord_to_xyz((row, col), grid_shape=grid_shape,
                                   origin=(x_offset, -0.1, 0.6), cell_size=cell_size)
                color = [1, 1, 1, 1]
                if (row, col) == goal_state:
                    color = [0, 1, 0, 1]
                elif wall_states and (row, col) in wall_states:
                    color = [0.4, 0.4, 0.4, 1]
                elif reward_map is not None:
                    r = reward_map[row, col]
                    if r is not None and r > 0:
                        color = [1, 1, 0, 1]

                vs = p.createVisualShape(p.GEOM_BOX,
                                         halfExtents=[box_half_size, box_half_size, 0.04],
                                         rgbaColor=color)
                p.createMultiBody(baseVisualShapeIndex=vs, basePosition=[pos[0], pos[1], pos[2] - 0.1])

###########################transform
    def get_ee_transform(self, robot_index):
        robot_id = self.robots[robot_index]
        ee_index = self.ee_link_index
        state = p.getLinkState(robot_id, ee_index, computeForwardKinematics=True)
        pos = state[4]
        orn = state[5]
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = pos
        return T

    def get_ee_position(self, robot_idx):
        state = p.getLinkState(self.robots[robot_idx], self.ee_link_index)
        return np.array(state[4])
    
    def get_ee_velocity(self, robot_index):
        robot_id = self.robots[robot_index]
        ee_index = self.ee_link_index 
        link_state = p.getLinkState(robot_id, ee_index, computeLinkVelocity=1,computeForwardKinematics=1)
        
        return np.array(link_state[6])