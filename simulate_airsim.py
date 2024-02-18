#!/usr/bin/env python3

"""
simulate_airsim.py
Author: Keiko Nagami

"""

import numpy as np
import airsim
import airsimdroneracinglab
import cv2
import time
import utils
import os
import copy
import math
import argparse
from scipy.spatial.transform import Rotation as R

import torch 
from visualizers import trajviz3D, plotmeasurements
from belief_plan_3D import TrajectoryOptimization
from estimate_3D import EKF
from train import NNLearning
from init_traj import ipopt_traj
from mpc_track import track_traj
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import matplotlib.cm as cm

class AirsimSim():
    def __init__(self):
        # AIRSIM THINGS
        self.airsim_client = airsimdroneracinglab.MultirotorClient()
        self.airsim_client.confirmConnection()

        self.airsim_client_camera = airsim.MultirotorClient()
        self.airsim_client_camera.confirmConnection()

        self.level_name = 'Soccer_Field_Easy' #'Soccer_Field_Medium'
        self.vehicle_name = 'drone_1'
        self.gate_count = 0
        self.lap_count = 2
        self.mu_y = 10.0

        self.load_level(self.level_name)
        self.airsim_client.race_tier = 1

        self.gate_names = self.get_gate_names()
        self.get_gatenum()

        # MPC THINGS
        self.num_steps = 200
        self.traj_length = 200
        self.xf = np.array([0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 0., -1.0, 0.])     # goal state
        self.s0 = 0.0001*np.identity(9)     # initial uncertainty
        self.dt = 0.02                      # time step
        self.case_num = 2                   # case_num = 4 predicted measurement with sigma grad, case_num = 2 max likelihood observation with sigma grad
 
        # load uncertainty map for visualization
        self.var_map = np.loadtxt("./images/varmaps/dronegate/large96k_test_x_map_inv.txt")

        # # Planner Sub-class
        self.trajopt_class = TrajectoryOptimization(self.dt, baseline=False, uncertainty_type="epistemic")
        self.trajopt_class.case_num = self.case_num
        self.trajopt_class.iterations = 2

        # # Estimator Sub-class
        self.ekf_class = EKF(self.trajopt_class.n, self.trajopt_class.m, self.trajopt_class.h, self.trajopt_class.dt)

        self.im_count = 0
        self.empty_image = False

        self.save_interval = 25
        
    def get_gate_names(self):
        # define gates
        if self.level_name == 'Soccer_Field_Easy':
            gate_names = ['Gate00', 'Gate01', 'Gate02', 'Gate03', 'Gate04', 'Gate05', 'Gate06', 'Gate07', 'Gate08', 'Gate09', 'Gate10_21', 'Gate11_23', 'Gate00']
        elif self.level_name == 'Soccer_Field_Medium':
            gate_names = ['Gate00', 'Gate01', 'Gate02', 'Gate03', 'Gate04', 'Gate05', 'Gate06', 'Gate07', 'Gate08', 'Gate09', 'Gate10', 'Gate11', 'Gate12', 'Gate13', 'Gate14', 'Gate15', 'Gate16', 'Gate17', 'Gate18', 'Gate19', 'Gate20', 'Gate21', 'Gate22', 'Gate23', 'Gate24']
        elif self.level_name == 'ZhangJiaJie_Medium':
            gate_names = ['Gate00', 'Gate01', 'Gate02', 'Gate03', 'Gate04', 'Gate05', 'Gate06', 'Gate07', 'Gate08', 'Gate09', 'Gate10', 'Gate11', 'Gate12', 'Gate13', 'Gate14', 'Gate15', 'Gate16', 'Gate17', 'Gate18', 'Gate19', 'Gate20', 'Gate21', 'Gate22', 'Gate23', 'Gate24']
        return gate_names

    def get_gt_image(self, global_pose):
        self.im_count += 1
        # MOVE CAMERA
        self.airsim_client.simSetVehiclePose(global_pose, True)
        request = [airsim.ImageRequest('fpv_cam', airsim.ImageType.Scene, False, False)]
        response = self.airsim_client_camera.simGetImages(request, vehicle_name=self.vehicle_name)

        img_rgb_1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)

        if np.linalg.norm(img_rgb) < 0.1:
            self.empty_image = True
            print("Empty Image")
            # return empty_image
            img_gray = None
        else:
            cv2.imwrite("./images/airsim/frame.jpg", img_rgb) #+ str(self.im_count) + ".jpg", img_rgb)
            img_gray = cv2.imread("./images/airsim/frame.jpg", cv2.IMREAD_GRAYSCALE) # + str(self.im_count) + ".jpg", cv2.IMREAD_GRAYSCALE)
        return img_gray

    def get_gatenum(self):
        gate_switch = False
        if self.mu_y < 1.0:
            self.gate_count += 1
            if self.gate_count == 13:
                self.gate_count = 1
                self.lap_count += 1
            print("Gate switch: ", self.gate_count)
            gate_switch = True

            # REINIT 
            self.ekf_class.sigma = self.s0

            # local previous gate to global
            pose_gt_local = (self.x[0:6] * self.trajopt_class.dronegate_bounds.detach().cpu().numpy()) + self.trajopt_class.dronegate_minimum.detach().cpu().numpy()
            pos_gt_local = pose_gt_local[0:3]
            eul_gt_local = pose_gt_local[3:6]
            vel_gt_local = self.x[6::]

            pose_et_local = (self.ekf_class.mu[0:6] * self.trajopt_class.dronegate_bounds.detach().cpu().numpy()) + self.trajopt_class.dronegate_minimum.detach().cpu().numpy()
            pos_et_local = pose_et_local[0:3]
            eul_et_local = pose_et_local[3:6]
            vel_et_local = self.ekf_class.mu[6::]
            
            rotation_roll180 = np.array([[1, 0, 0],[0, np.cos(np.pi),-np.sin(np.pi)],[0, np.sin(np.pi), np.cos(np.pi)]])

            # previous gate
            gate_position_airsim = np.array([self.gate_pose.position.x_val, self.gate_pose.position.y_val, self.gate_pose.position.z_val])
            gate_orientation_airsim = self.getRotationMatrix_quat(self.gate_pose.orientation)

            gate_position_zprev = rotation_roll180 @ gate_position_airsim

            gate_position_global = gate_position_zprev - np.array([0., 0., gate_position_zprev[2]])
            gate_orientation_global = rotation_roll180 @ gate_orientation_airsim @ rotation_roll180

            # global
            pos_gt_global = gate_orientation_global @ pos_gt_local + gate_position_global
            eul_gt_global = self.getGateEulerAngles(gate_orientation_global @ self.getRotationMatrix_euler(eul_gt_local))
            vel_gt_global = gate_orientation_global @ vel_gt_local
            pos_et_global = gate_orientation_global @ pos_et_local + gate_position_global
            eul_et_global = self.getGateEulerAngles(gate_orientation_global @ self.getRotationMatrix_euler(eul_et_local))
            vel_et_global = gate_orientation_global @ vel_et_local

            # print("pos_gt_global: ", pos_gt_global)

            # new gate
            self.gate_pose = self.airsim_client.simGetObjectPose(self.gate_names[self.gate_count])

            gate_position_airsim = np.array([self.gate_pose.position.x_val, self.gate_pose.position.y_val, self.gate_pose.position.z_val])
            gate_orientation_airsim = self.getRotationMatrix_quat(self.gate_pose.orientation)

            gate_position_zprev = rotation_roll180 @ gate_position_airsim

            gate_position_global = gate_position_zprev - np.array([0., 0., gate_position_zprev[2]])
            gate_orientation_global = rotation_roll180 @ gate_orientation_airsim @ rotation_roll180

            # rotate into the new frame
            pos_gt_local_new = -gate_orientation_global.T @ (gate_position_global - pos_gt_global)
            eul_gt_local_new = self.getGateEulerAngles(gate_orientation_global.T @ self.getRotationMatrix_euler(eul_gt_global))
            vel_gt_local_new = -gate_orientation_global.T @ (-vel_gt_global)
            pos_et_local_new = -gate_orientation_global.T @ (gate_position_global - pos_et_global)
            eul_et_local_new = self.getGateEulerAngles(gate_orientation_global.T @ self.getRotationMatrix_euler(eul_et_global))
            vel_et_local_new = -gate_orientation_global.T @ (-vel_et_global)

            pose_gt_local_new = (np.hstack((pos_gt_local_new, eul_gt_local_new)) - self.trajopt_class.dronegate_minimum.detach().cpu().numpy()) / self.trajopt_class.dronegate_bounds.detach().cpu().numpy()
            self.x[0:3] = pose_gt_local_new[0:3]
            self.x[3:6] = pose_gt_local_new[3:6]
            self.x[6::] = vel_gt_local_new
            
            pose_et_local_new = (np.hstack((pos_et_local_new, eul_et_local_new)) - self.trajopt_class.dronegate_minimum.detach().cpu().numpy()) / self.trajopt_class.dronegate_bounds.detach().cpu().numpy()
            self.ekf_class.mu[0:3] = pose_et_local_new[0:3]
            self.ekf_class.mu[3:6] = pose_et_local_new[3:6]
            self.ekf_class.mu[6::] = vel_et_local_new

            # check if poor initial estimate causes gate miss
            self.mu_y = pos_et_local_new[1]
                        
        else:
            self.gate_pose = self.airsim_client.simGetObjectPose(self.gate_names[self.gate_count])

        return gate_switch

    """
    def getRotationMatrix(airsim_vector_orientation):
        gets the rotation matrix corresponding to the quaternionr inputs
    """
    def getRotationMatrix_quat(self, airsim_vector_orientation):
        q0 = airsim_vector_orientation.w_val
        q1 = airsim_vector_orientation.x_val
        q2 = airsim_vector_orientation.y_val
        q3 = airsim_vector_orientation.z_val
        rot_matrix_obj2global = np.array([[1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
                                        [2*(q1*q2 + q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
                                        [2*(q1*q3-q0*q2), 2*(q1*q0+q2*q3), 1-2*(q1**2+q2**2)]])
        return rot_matrix_obj2global

    """
    def quat2euler(quat):
        gets euler angles from quaternions
    """
    def quat2euler(self, airsim_quat):
        qw = quat.w_val
        qx = quat.x_val
        qy = quat.y_val
        qz = quat.z_val
        roll = np.arctan2(2*(qw*qx + qy*qz) , 1-2*(qx**2 + qy**2))
        pitch = np.arcsin(2*(qw*qy - qz*qx))
        yaw = np.arctan2(2*(qw*qz + qx*qy) , 1-2*(qy**2 + qz**2))
        euler_angles = np.array([roll, pitch, yaw])
        return euler_angles

    """
    def getRotationMatrix_euler(euler_angles):
        gets rotation matrix corresponding to the euler angle inputs
    """
    def getRotationMatrix_euler(self, euler_angles):
        phi = euler_angles[0] # roll
        theta = euler_angles[1] # pitch
        psi = euler_angles[2] # yaw
        # yaw
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],[np.sin(psi), np.cos(psi), 0],[0, 0, 1]])
        # pitch
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],[0, 1, 0],[-np.sin(theta), 0, np.cos(theta)]])
        # roll
        Rx = np.array([[1, 0, 0],[0, np.cos(phi),-np.sin(phi)],[0, np.sin(phi), np.cos(phi)]])

        rot_matrix = np.linalg.multi_dot([Rz,Ry,Rx])
        return rot_matrix

    def getGateEulerAngles(self, rotation_mtx):
        roll = math.atan2(rotation_mtx[2,1],rotation_mtx[2,2])
        pitch = math.atan2(-rotation_mtx[2,0],math.sqrt(rotation_mtx[2,1]**2 + rotation_mtx[2,2]**2))
        yaw = math.atan2(rotation_mtx[1,0], rotation_mtx[0,0])
        return roll, pitch, yaw  # radians

    def get_airsim_pose(self, local_pose):
        # print("get_airsim_pose1: ", self.x)
        # unnormalize
        local_pose = (local_pose * self.trajopt_class.dronegate_bounds.detach().cpu().numpy()) + self.trajopt_class.dronegate_minimum.detach().cpu().numpy()
        # print("get_airsim_pose2: ", self.x)

        # print("airsim pose unnormalized: ", local_pose)
        rotation_roll180 = np.array([[1, 0, 0],[0, np.cos(np.pi),-np.sin(np.pi)],[0, np.sin(np.pi), np.cos(np.pi)]])
        
        # GATE DEFS
        gate_position_airsim = np.array([self.gate_pose.position.x_val, self.gate_pose.position.y_val, self.gate_pose.position.z_val])
        gate_orientation_airsim = self.getRotationMatrix_quat(self.gate_pose.orientation)

        gate_position_zprev = rotation_roll180 @ gate_position_airsim

        gate_position_global = gate_position_zprev - np.array([0., 0., gate_position_zprev[2]])
        gate_orientation_global = rotation_roll180 @ gate_orientation_airsim @ rotation_roll180

        # POSITION
        quad_position_local = local_pose[0:3]
        quad_position_global = gate_orientation_global @ quad_position_local + gate_position_global
        quad_position_zpost = quad_position_global + np.array([0., 0., gate_position_zprev[2]])
        quad_position_airsim = rotation_roll180 @ quad_position_zpost

        # ORIENTATION
        quad_orientation_local = self.getRotationMatrix_euler(local_pose[3:6])
        quad_orientation_global = np.dot(gate_orientation_global, quad_orientation_local)
        quad_orientation_airsim = rotation_roll180 @ quad_orientation_global @ rotation_roll180
        quad_quat_airsim = R.from_matrix(quad_orientation_airsim).as_quat()

        quad_position_airsim = airsimdroneracinglab.Vector3r(quad_position_airsim[0], quad_position_airsim[1], quad_position_airsim[2])
        quad_quat_airsim = airsimdroneracinglab.Quaternionr(quad_quat_airsim[0], quad_quat_airsim[1], quad_quat_airsim[2], quad_quat_airsim[3])
        quad_pose_airsim = airsimdroneracinglab.Pose(quad_position_airsim, quad_quat_airsim)
        return quad_pose_airsim
    
    # loads desired level
    def load_level(self, level_name, sleep_sec = 2.0):
        self.level_name = level_name
        self.airsim_client.simLoadLevel(self.level_name)
        self.airsim_client.confirmConnection() # failsafe
        self.airsim_client_camera.confirmConnection()
        time.sleep(sleep_sec) # let the environment load completely

    def quad_dynamics(self, u):
        # clip u
        u = np.clip(u, self.trajopt_class.u_min.detach().cpu().numpy().reshape((4,)), self.trajopt_class.u_max.detach().cpu().numpy().reshape((4,)))
        # unnormalize
        x = copy.deepcopy(self.x)
        x[0:6] = (x[0:6] * self.trajopt_class.dronegate_bounds.detach().cpu().numpy()) + self.trajopt_class.dronegate_minimum.detach().cpu().numpy()

        # define constants
        grav = np.array([0., 0., 9.81])
        zb = np.array([0, 0, 1])

        Rx = np.array([[1, 0, 0],
                        [0, math.cos(x[3]), -math.sin(x[3])],
                        [0, math.sin(x[3]), math.cos(x[3])]])
        Ry = np.array([[math.cos(x[4]), 0, math.sin(x[4])],
                        [0, 1, 0],
                        [-math.sin(x[4]), 0, math.cos(x[4])]])
        Rz = np.array([[math.cos(x[5]), -math.sin(x[5]), 0],
                        [math.sin(x[5]), math.cos(x[5]), 0],
                        [0, 0, 1]])

        body2world_mtx = Rz @ Ry @ Rx
        a = -grav + np.dot(body2world_mtx, u[0]*zb)

        body2euler_rate_mtx = np.array([[1, math.sin(x[3])*math.tan(x[4]), math.cos(x[3])*math.tan(x[4])],
                                        [0, math.cos(x[3]), -math.sin(x[3])], 
                                        [0, math.sin(x[3])*(1./math.cos(x[4])), math.cos(x[3])*(1./math.cos(x[4]))]])

        rand_p  = np.random.normal(0.0, 0.003)
        rand_th = np.random.normal(0.0, 0.001) 
        rand_v  = np.random.normal(0.0, 0.001)

        p  = x[0:3] + self.dt*x[6:9] + rand_p
        th = x[3:6] + self.dt*(body2euler_rate_mtx @ u[1::]) + rand_th
        v  = x[6:9] + self.dt*a + rand_v
        x_new = np.hstack((p,th,v))
        # self.gt_y = p[1]

        # normalize
        x_new[0:6] = (x_new[0:6] - self.trajopt_class.dronegate_minimum.detach().cpu().numpy()) / self.trajopt_class.dronegate_bounds.detach().cpu().numpy()
        return x_new

    def save_point(self,i, gate_count, u_opt, traj_u, traj_gt, traj_est, traj_z, traj_Q, traj_S, traj_R, X_nominal):
        np.savetxt("./empty_save/u_opt_" + str(self.lap_count) + "_" + str(gate_count) + "_" + str(i) + ".txt", u_opt)
        np.savetxt("./empty_save/traj_u_" + str(self.lap_count) + "_" + str(gate_count) + "_" + str(i) + ".txt", traj_u)

        np.savetxt("./empty_save/x_now_" + str(self.lap_count) + "_" + str(gate_count) + "_" + str(i) + ".txt", self.x)
        np.savetxt("./empty_save/mu_now_" + str(self.lap_count) + "_" + str(gate_count) + "_" + str(i) + ".txt", self.ekf_class.mu)
        np.savetxt("./empty_save/s_now_" + str(self.lap_count) + "_" + str(gate_count) + "_" + str(i) + ".txt", self.ekf_class.sigma)
        
        np.savetxt("./empty_save/traj_gt_" + str(self.lap_count) + "_" + str(gate_count) + "_" + str(i) + ".txt", traj_gt)
        np.savetxt("./empty_save/traj_est_" + str(self.lap_count) + "_" + str(gate_count) + "_" + str(i) + ".txt", traj_est)
        np.savetxt("./empty_save/traj_z_" + str(self.lap_count) + "_"  + str(gate_count) + "_" + str(i) + ".txt", traj_z)

        np.savetxt("./empty_save/traj_Q_" + str(self.lap_count) + "_" + str(gate_count) + "_" + str(i) + ".txt", traj_Q)
        np.savetxt("./empty_save/traj_S_" + str(self.lap_count) + "_" + str(gate_count) + "_" + str(i) + ".txt", traj_S)
        np.savetxt("./empty_save/traj_R_" + str(self.lap_count) + "_" + str(gate_count) + "_" + str(i) + ".txt", traj_R)

        np.savetxt("./empty_save/gatect_alpha_imct_pconst_" + str(self.lap_count) + "_" + str(gate_count) + "_" + str(i) + ".txt", np.array([self.gate_count, self.trajopt_class.alpha, self.im_count-1, self.trajopt_class.pconst]))
        np.savetxt("./empty_save/X_nominal_" + str(self.lap_count) + "_" + str(gate_count) + "_" + str(i) + ".txt", X_nominal)
        if self.trajopt_class.baseline == False:
            torch.save(self.trajopt_class.optimizer.state_dict(), "./empty_save/opt_state_dict_" + str(self.lap_count) + "_" + str(gate_count) + "_" + str(i) + ".pth")


    def runAirsimSim(self, args):
        print("restart: ", args.restart)
        if args.restart == False:
            self.x0 = np.array([0.4, 0.4, 0.55, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
            self.x  = self.x0.copy()
            self.ekf_class.mu    = self.x0 + np.random.normal(0.0, 0.001, size=(9,))
            self.ekf_class.sigma = self.s0

        elif args.restart == True:
            self.gate_count = args.gate_count
            self.x0   = np.loadtxt("./empty_save/x_now_" + str(self.lap_count) + "_" + str(self.gate_count) + "_" + str(args.iteration) + ".txt")
            self.x    = self.x0.copy()
             
            self.ekf_class.mu    = np.loadtxt("./empty_save/mu_now_" + str(self.lap_count) + "_" + str(self.gate_count) + "_" + str(args.iteration) + ".txt")
            self.ekf_class.sigma = np.loadtxt("./empty_save/s_now_" + str(self.lap_count) + "_" + str(self.gate_count) + "_" + str(args.iteration) + ".txt")
            self.mu_y =  (self.ekf_class.mu[1] * self.trajopt_class.dronegate_bounds[1]) + self.trajopt_class.dronegate_minimum[1]
            if self.mu_y < 3.0:
                print("recede goal")
                self.xf[1] = 0.0
                recede_count = 1
            else:
                recede_count = 0
            self.u_init = np.loadtxt("./empty_save/u_opt_" + str(self.lap_count) + "_" + str(self.gate_count) + "_" + str(args.iteration) + ".txt")
            traj_u      = np.loadtxt("./empty_save/traj_u_" + str(self.lap_count) + "_" + str(self.gate_count) + "_" + str(args.iteration) + ".txt")
            u_opt       = self.u_init.copy()
            u           = self.u_init.copy()

            traj_est = np.loadtxt("./empty_save/traj_est_" + str(self.lap_count) + "_" + str(self.gate_count) + "_" + str(args.iteration) + ".txt")
            traj_gt  = np.loadtxt("./empty_save/traj_gt_" + str(self.lap_count) + "_" + str(self.gate_count) + "_" + str(args.iteration) + ".txt")
            traj_z   = np.loadtxt("./empty_save/traj_z_" + str(self.lap_count) + "_" + str(self.gate_count) + "_" + str(args.iteration) + ".txt")

            traj_Q = np.loadtxt("./empty_save/traj_Q_" + str(self.lap_count) + "_" + str(self.gate_count) + "_" + str(args.iteration) + ".txt")
            traj_S  = np.loadtxt("./empty_save/traj_S_" + str(self.lap_count) + "_" + str(self.gate_count) + "_" + str(args.iteration) + ".txt")
            traj_R   = np.loadtxt("./empty_save/traj_R_" + str(self.lap_count) + "_" + str(self.gate_count) + "_" + str(args.iteration) + ".txt")

            X_nominal = np.loadtxt("./empty_save/X_nominal_" + str(self.lap_count) + "_" + str(self.gate_count) + "_" + str(args.iteration) + ".txt")
            traj_length    = self.u_init.shape[1] + 1
            self.num_steps = self.u_init.shape[1] + 1
            
            other_params = np.loadtxt("./empty_save/gatect_alpha_imct_pconst_" + str(self.lap_count) + "_" + str(self.gate_count) + "_" + str(args.iteration) + ".txt")
            self.get_gatenum()

            self.trajopt_class.alpha = other_params[1]
            self.im_count = int(other_params[2])
            self.trajopt_class.pconst = other_params[3]
            
        while self.gate_count < 13:
            if args.restart == False:
                mu_phys = copy.deepcopy(self.ekf_class.mu)
                mu_phys[0:6] = (self.ekf_class.mu[0:6] * self.trajopt_class.dronegate_bounds.detach().cpu().numpy()) + self.trajopt_class.dronegate_minimum.detach().cpu().numpy()
                self.xf[1] = 0.2
                xf = copy.deepcopy(self.xf)
                xf[0:6] = (xf[0:6] * self.trajopt_class.dronegate_bounds.detach().cpu().numpy()) + self.trajopt_class.dronegate_minimum.detach().cpu().numpy()

                traj_est = self.ekf_class.mu.reshape((-1, 1))
                traj_gt  = self.x.reshape((-1, 1))
                traj_Q = np.diag(self.ekf_class.Q)
                traj_S = np.diag(self.ekf_class.sigma)
                if self.gate_count == 0:
                    traj_length = 100
                else:
                    traj_length = 200 # self.traj_length

                ipopt_u, ipopt_x = ipopt_traj(mu_phys, xf, traj_length, baseline=False)
                self.u_init = np.zeros((self.trajopt_class.m, traj_length-1))
                X = np.zeros((self.trajopt_class.n, traj_length))

                self.u_init[:,:] = ipopt_u[:,:] 
                X[:,:] = ipopt_x[:,:]
                u_opt = self.u_init.copy()
                
                self.trajopt_class.max_iters  = 100
                self.trajopt_class.pconst = 1.e-5
                self.trajopt_class.alpha = 2.e-3
                i = 0
                recede_count = 0
                opt_state_dict_fname = "./opt_state_dict.pth"

            elif args.restart == True:

                i = args.iteration + 1
                opt_state_dict_fname = "./empty_save/opt_state_dict_"  + str(self.gate_count) + "_" + str(args.iteration) + ".pth"

            num_iters = 0
            final_cost = 10.
            time_array = []
            while self.mu_y > 0.5:
                # compute action from current state
                self.trajopt_class.init_traj(self.ekf_class.mu, self.ekf_class.sigma, self.xf, traj_length)

                opt_state_dict = None

                if (final_cost >= 1000.) or ((i==0) and (self.trajopt_class.baseline==True)):
                    mu_phys = copy.deepcopy(self.ekf_class.mu)
                    mu_phys[0:6] = (self.ekf_class.mu[0:6] * self.trajopt_class.dronegate_bounds.detach().cpu().numpy()) + self.trajopt_class.dronegate_minimum.detach().cpu().numpy()

                    xf = copy.deepcopy(self.xf)
                    xf[0:6] = (xf[0:6] * self.trajopt_class.dronegate_bounds.detach().cpu().numpy()) + self.trajopt_class.dronegate_minimum.detach().cpu().numpy()
                    if i == 0:
                        ipopt_u, ipopt_x = ipopt_traj(mu_phys, xf, traj_length, X, u_opt.T.flatten(), baseline=False)
                    else:
                        ipopt_u, ipopt_x = ipopt_traj(mu_phys, xf, traj_length, X[:, 1::], u_opt.T.flatten(), baseline=False)
                    u = np.zeros((self.trajopt_class.m, traj_length-1))
                    X = np.zeros((self.trajopt_class.n, traj_length))

                    u[:,:] = ipopt_u[:,:]
                    X[:,:] = ipopt_x[:,:]
                    if i == 0:
                        X_nominal = copy.deepcopy(X)
                    else:
                        X_nominal = np.column_stack((X_nominal[:,0:i], X))
                    u = torch.from_numpy(u)
                    final_cost = 10.
                elif (i == 0) and (self.trajopt_class.baseline == False):
                    u, X_nominal, S, num_iters, final_cost = self.trajopt_class.solve_opt(i,u_opt,opt_state_dict)
                    X = copy.deepcopy(X_nominal)
                else:
                    mu_phys = copy.deepcopy(self.ekf_class.mu)
                    mu_phys[0:6] = (self.ekf_class.mu[0:6] * self.trajopt_class.dronegate_bounds.detach().cpu().numpy()) + self.trajopt_class.dronegate_minimum.detach().cpu().numpy()
                    plan_start_time = time.time()
                    mpc_u, mpc_x = track_traj(mu_phys, traj_length, X_nominal[:,i::], u_opt)
                    plan_end_time = time.time()
                    plan_time = plan_end_time - plan_start_time
                    time_array.append(plan_time)
                    u = np.zeros((self.trajopt_class.m, traj_length-1))
                    X = np.zeros((self.trajopt_class.n, traj_length))

                    u[:,:] = mpc_u[:,:]
                    X[:,:] = mpc_x[:,:]

                    u = torch.from_numpy(u)

                u = u.detach().cpu().numpy()

                # apply action
                self.x = self.quad_dynamics(u[:,0])
                u_opt = u[:, 1::] #np.column_stack((u[:, 1::], u[:, -1]))

                # take a measurement
                gate_switch = self.get_gatenum()                     # check if we're still in front of the same gate
                if gate_switch:
                    break

                self.airsim_client.simPause(False)
                quad_airsim_pose = self.get_airsim_pose(self.x[0:6])     # convert local pose to airsim pose
                z_image = self.get_gt_image(quad_airsim_pose)
                self.airsim_client.simPause(True)

                # if the image is empty
                if self.empty_image == True:
                    break

                z_image = cv2.resize(z_image, (64,64))
                z_image = z_image / 255.0

                image_tensor = torch.from_numpy(z_image).to(self.trajopt_class.learning_class.device)
                image_tensor_shape = image_tensor.size()
                image_tensor = torch.reshape(image_tensor,(1, 1,image_tensor_shape[0],image_tensor_shape[1] ))
                image_tensor = image_tensor.float()
                if self.trajopt_class.baseline == False:
                    z = self.trajopt_class.learning_class.encoder(image_tensor)
                elif self.trajopt_class.baseline == True:
                    z, Rb  = self.trajopt_class.learning_class.encoder(image_tensor)
                z = z.squeeze().detach().cpu().numpy()
                
                # unnormalize
                z = (z * self.trajopt_class.dronegate_bounds.detach().cpu().numpy()) + self.trajopt_class.dronegate_minimum.detach().cpu().numpy()
                
                # estimate current state
                if (self.mu_y <= 2.0) or (self.mu_y >= 7.0):
                    GPS = True
                    z = (self.x[0:6] * self.trajopt_class.dronegate_bounds.detach().cpu().numpy()) + self.trajopt_class.dronegate_minimum.detach().cpu().numpy() 
                    pos_rand = np.random.normal(0, 0.05, size=(3,))
                    eul_rand = np.random.normal(0, 0.02, size=(3,))
                    z = z + np.hstack((pos_rand, eul_rand))
                else:
                    GPS = False
                if self.trajopt_class.baseline == False:
                    __, __, R = self.ekf_class.ekf(u[:,0], z, self.trajopt_class.learning_class, GPS)
                elif self.trajopt_class.baseline == True:
                    R = np.diag(Rb.squeeze().detach().cpu().numpy()) 
                    __, __, __ = self.ekf_class.ekf(u[:,0], (z,R), self.trajopt_class.learning_class, GPS)
                self.mu_y = (self.ekf_class.mu[1]*self.trajopt_class.dronegate_bounds[1]) + self.trajopt_class.dronegate_minimum[1]
                traj_gt = np.column_stack((traj_gt, self.x.reshape((-1,1))))
                traj_est = np.column_stack((traj_est, self.ekf_class.mu.reshape((-1,1))))

                traj_Q = np.column_stack((traj_Q, np.diag(self.ekf_class.Q)))
                traj_S = np.column_stack((traj_S, np.diag(self.ekf_class.sigma)))

                if i == 0:
                    traj_z = z.copy()
                    traj_R = np.diag(R)
                    traj_u = u[:,0]
                else:
                    traj_z = np.column_stack((traj_z, z.reshape((-1,1))))
                    traj_R = np.column_stack((traj_R, np.diag(R)))
                    traj_u = np.column_stack((traj_u, u[:,0]))

                if traj_length > self.trajopt_class.kf:
                    traj_length -= 1
                else:
                    u_opt = np.column_stack((u[:,1::], u[:,-1]))  
                    X = np.column_stack((X, X[:,-1]))
                    X_nominal = np.column_stack((X_nominal, X_nominal[:,-1]))
                args.restart = False
                opt_state_dict_fname = "./opt_state_dict.pth"

                if self.mu_y < 3.0:
                    self.xf[1] = 0.0
                    # self.xf[7] = -1.5
                    if recede_count == 0:
                        final_cost = 1001.
                        recede_count = 1
                        print("recede goal")
                   
                i += 1

            if self.empty_image == True:
                break
        self.lap_count += 1
        self.gate_count = 1
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gate_num', dest = 'gate_count', type = int, default = 2)
    parser.add_argument('--iteration', dest = 'iteration', type = int, default = 0)
    parser.add_argument('--restart', dest = 'restart', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    airsimsim_class = AirsimSim()
    airsimsim_class.runAirsimSim(args)
