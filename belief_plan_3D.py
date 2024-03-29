import numpy as np
from train_baseline import NNLearning as NNBaseline
from train import NNLearning

import time
import torch
import matplotlib.pyplot as plt
from visualizers import trajviz3D, plotcost, plotplan
from generate_data import loadImage
import matplotlib.cm as cm
import matplotlib.patches as patches 
# from test import varmap
from torch.distributions import multivariate_normal
from torch.autograd import Variable
import torch.optim as optim
from A_calc_torch import a_calc
from init_traj import ipopt_traj
import copy

class TrajectoryOptimization():
    def __init__(self, dt, baseline, uncertainty_type):
        # problem parameters
        self.m = 4
        self.n = 9
        self.h = 6
        self.dt = dt
        self.case_num = 2
        self.kf = 50
        self.max_iters = 2000
        self.max_cost = 1.e-4
        # perception networks
        self.baseline = baseline
        self.uncertainty_type = uncertainty_type
        if self.uncertainty_type == "epistemic":
            if self.baseline == True:
                self.learning_class = NNBaseline(dataset_folder ="./dronegate",  model_name = "large_96kb_debug", save_name="_checkpt3", checkpoint = False, checkpoint_name = "_checkpt4")
                self.learning_class.encoder = self.learning_class.loadModel("./dronegate/save/large_96kb_debug/encoder_weights_checkpt3.pth")
            elif self.baseline == False:
                self.learning_class = NNLearning(dataset_folder ="./dronegate",  model_name = "large_96k", save_name="_checkpt", checkpoint = False, checkpoint_name = "_checkpt2")
                self.learning_class.encoder = self.learning_class.loadModel("./dronegate/save/large_96k/encoder_weights_checkpt2.pth")
            self.learning_class.decoder = self.learning_class.loadModel("./dronegate/save/large_24k/decoder_weights_checkpt.pth")
        elif self.uncertainty_type == "aleatoric":
            if self.baseline == True:
                self.learning_class = NNBaseline(dataset_folder ="./dronegate",  model_name = "aleatoric_48kb_noisier", save_name="_checkpt2", checkpoint = False, checkpoint_name = "_checkpt3")
                self.learning_class.encoder = self.learning_class.loadModel("./dronegate/save/aleatoric_48kb_noisier/encoder_weights_checkpt2.pth")
            elif self.baseline == False:
                self.learning_class = NNLearning(dataset_folder = "./dronegate", model_name="aleatoric_48k_noisier", save_name="", checkpoint=True, checkpoint_name="_checkpt")
                self.learning_class.encoder = self.learning_class.loadModel("./dronegate/save/aleatoric_48k_noisier/encoder_weights_checkpt.pth")
            self.learning_class.decoder = self.learning_class.loadModel("./dronegate/save/aleatoric_48k_noisier/decoder_weights_checkpt.pth")
        self.learning_class.encoder.eval()
        self.learning_class.decoder.eval()
        # trajectory optimizer
        self.optimizer = None 
        self.scheduler = None

        self.alpha = 1.e-2
        self.gamma_lr = 0.99
        self.gamma_pn = 0.99
        self.varmax = 1.e-2 
        self.pconst = 1.e-5 
        # dynamics
        self.grav = torch.tensor([0., 0., 9.81]).to(self.learning_class.device)
        self.dronegate_minimum = torch.tensor([-5., 0., -2.5, -np.pi/2., -np.pi/2., -np.pi]).to(self.learning_class.device)
        self.dronegate_bounds  = torch.tensor([10., 10., 5., np.pi, np.pi, np.pi]).to(self.learning_class.device)
        self.dronegate_minimum_2D = torch.reshape(self.dronegate_minimum, (1,6))
        self.dronegate_bounds_2D = torch.reshape(self.dronegate_bounds, (1,6))
        self.H = torch.zeros((self.h, self.n)).to(self.learning_class.device)
        self.H[0:6,0:6] = torch.eye(self.h)
        self.zb = torch.tensor([0, 0, 1]).to(self.learning_class.device)
        self.torchpi = self.numpy2torch(np.array([np.pi]))
        self.Sq = 0.0001*self.numpy2torch(np.identity(self.n))
        self.Ic = self.numpy2torch(np.identity(self.n))

        self.u_min = torch.tensor([[0.0],   [-3.5], [-3.5], [-2.0]]).to(self.learning_class.device)
        self.u_max = torch.tensor([[9.81*2], [3.5],  [3.5],  [2.0]]).to(self.learning_class.device)
        self.logging = False

    def init_traj(self, p0, s0, xf, K):
        self.k = K
        
        self.xf = xf
        self.xf_tensor = self.numpy2torch(np.tile(xf,self.k))
        
        self.p0 = p0
        self.s0 = s0
        
        # state cost
        self.Q_tensor = self.numpy2torch(np.zeros((self.n*self.k, self.n*self.k)))
        self.Q_tensor[self.k*self.n-9, self.k*self.n-9] = 1.0 # x
        self.Q_tensor[self.k*self.n-8, self.k*self.n-8] = 1.0 # y
        self.Q_tensor[self.k*self.n-7, self.k*self.n-7] = 1.0 # z
        self.Q_tensor[self.k*self.n-6, self.k*self.n-6] = 0.1 # phi
        self.Q_tensor[self.k*self.n-5, self.k*self.n-5] = 0.1 # th
        self.Q_tensor[self.k*self.n-4, self.k*self.n-4] = 0.1 # psi

        # control cost
        a = 0.1*np.array([0.1, 1., 1., 1.]) # thrust roll pitch yaw
        b = np.tile(a,self.k-1)
        self.R_tensor = self.numpy2torch(np.diag(b))
        
        self.cost1list = []
        self.cost2list = []
        self.cost3list = []
        self.costlist = []

    def numpy2torch(self, var):
        var_torch = torch.from_numpy(var).float().to(self.learning_class.device)
        var_torch.requires_grad_(requires_grad=False)
        return var_torch

    def quad_dynamics_torch(self, p_no_norm, u_now):

        Rx1 = torch.tensor([[1, 0, 0],[0, 0, 0],[0, 0, 0]]).to(self.learning_class.device)
        Rx2 = torch.sin(p_no_norm[3])*torch.tensor([[0, 0, 0],[0, 0,-1],[0, 1, 0]]).to(self.learning_class.device)
        Rx3 = torch.cos(p_no_norm[3])*torch.tensor([[0, 0, 0],[0, 1, 0],[0, 0, 1]]).to(self.learning_class.device)
        Rx = Rx1 + Rx2 + Rx3

        Ry1 = torch.tensor([[0, 0, 0],[0, 1, 0],[0, 0, 0]]).to(self.learning_class.device)
        Ry2 = torch.sin(p_no_norm[4])*torch.tensor([[0, 0, 1],[0, 0, 0],[-1, 0, 0]]).to(self.learning_class.device)
        Ry3 = torch.cos(p_no_norm[4])*torch.tensor([[1, 0, 0],[0, 0, 0],[ 0, 0, 1]]).to(self.learning_class.device)
        Ry = Ry1 + Ry2 + Ry3

        Rz1 = torch.tensor([[0, 0, 0],[0, 0, 0],[0, 0, 1]]).to(self.learning_class.device)
        Rz2 = torch.sin(p_no_norm[5])*torch.tensor([[0,-1, 0],[1, 0, 0],[0, 0, 0]]).to(self.learning_class.device)
        Rz3 = torch.cos(p_no_norm[5])*torch.tensor([[1, 0, 0],[0, 1, 0],[0, 0, 0]]).to(self.learning_class.device)
        Rz = Rz1 + Rz2 + Rz3

        # Rx = torch.tensor([[1, 0, 0],[0, torch.cos(p_no_norm[3]), -torch.sin(p_no_norm[3])],[0, torch.sin(p_no_norm[3]), torch.cos(p_no_norm[3])]]).to(self.learning_class.device)
        # Ry = torch.tensor([[torch.cos(p_no_norm[4]), 0, torch.sin(p_no_norm[4])],[0, 1, 0],[-torch.sin(p_no_norm[4]), 0, torch.cos(p_no_norm[4])]]).to(self.learning_class.device)
        # Rz = torch.tensor([[torch.cos(p_no_norm[5]), -torch.sin(p_no_norm[5]), 0],[torch.sin(p_no_norm[5]), torch.cos(p_no_norm[5]), 0],[0, 0, 1]]).to(self.learning_class.device)
        body2world_mtx = torch.matmul(Rz, torch.matmul(Ry, Rx))
        
        a_pred = -self.grav + torch.matmul(body2world_mtx, u_now[0]*self.zb)

        body2euler_rate1 = torch.tensor([[1, 0, 0],[0, 0, 0],[0, 0, 0]]).to(self.learning_class.device)
        body2euler_rate2 = torch.sin(p_no_norm[3])*torch.tan(p_no_norm[4])*torch.tensor([[0, 1, 0],[0, 0, 0],[0, 0, 0]]).to(self.learning_class.device)
        body2euler_rate3 = torch.cos(p_no_norm[3])*torch.tan(p_no_norm[4])*torch.tensor([[0, 0, 1],[0, 0, 0],[0, 0, 0]]).to(self.learning_class.device)
        body2euler_rate4 = torch.cos(p_no_norm[3])*torch.tensor([[0, 0, 0],[0, 1, 0],[0, 0, 0]]).to(self.learning_class.device)
        body2euler_rate5 = torch.sin(p_no_norm[3])*torch.tensor([[0, 0, 0],[0, 0,-1],[0, 0, 0]]).to(self.learning_class.device)
        body2euler_rate6 = torch.sin(p_no_norm[3])*(1./torch.cos(p_no_norm[4]))*torch.tensor([[0, 0, 0],[0, 0, 0],[0, 1, 0]]).to(self.learning_class.device)
        body2euler_rate7 = torch.cos(p_no_norm[3])*(1./torch.cos(p_no_norm[4]))*torch.tensor([[0, 0, 0],[0, 0, 0],[0, 0, 1]]).to(self.learning_class.device)

        body2euler_rate_mtx = body2euler_rate1 + body2euler_rate2 + body2euler_rate3 + body2euler_rate4 + body2euler_rate5 + body2euler_rate6 + body2euler_rate7
        
        p_pred_velocity = p_no_norm[6:9] + self.dt*a_pred
        p_pred_position = p_no_norm[0:3] + self.dt*p_no_norm[6::]
        p_pred_orientation_init = p_no_norm[3:6] + self.dt*torch.matmul(body2euler_rate_mtx, u_now[1::])

        roll = p_pred_orientation_init[0].view(1) 
        pitch = p_pred_orientation_init[1].view(1)
        yaw = p_pred_orientation_init[2].view(1)

        if abs(roll) > self.torchpi:    
            print("roll out of bounds: ", roll)

        if abs(pitch) > self.torchpi:    
            print("pitch out of bounds: ", pitch)

        if abs(yaw + self.torchpi/2.) > self.torchpi:    
            print("yaw out of bounds: ", yaw)

        p_pred_orientation = torch.cat((roll, pitch, yaw))
        p_pred = torch.cat((p_pred_position, p_pred_orientation, p_pred_velocity))
        return p_pred

    def update_step_tensor(self, i, p_now, s_now, u_now, dt, dynamics_model = "quadrotor"):
        N = 50 # number of samples

        p_pose_no_norm = (torch.matmul(self.H, p_now) * self.dronegate_bounds) + self.dronegate_minimum
        p_no_norm = torch.cat((p_pose_no_norm, p_now[6::]))
        
        # unnormalize
        A_tensor = a_calc(p_no_norm, u_now, self.dt, 1.0, self.learning_class)

        if dynamics_model == "quadrotor":
            # PREDICT
            p_pred = self.quad_dynamics_torch(p_no_norm, u_now)
            
        elif dynamics_model == "single_integrator":

            p_pred_velocity = p_now[6:9] + dt*u_now[0:3]
            p_pred_position = p_no_norm[0:3] + dt*p_pred_velocity
            p_pred_orientation = p_no_norm[3::] + dt*u_now[3]*torch.tensor([0,0,1]).to(self.learning_class.device)
            p_pred = torch.cat((p_pred_position, p_pred_orientation, p_pred_velocity))
        
        s_pred = torch.matmul(A_tensor, torch.matmul(s_now, torch.t(A_tensor))) + self.Sq
        
        # sample
        rand_vec = torch.randn(N, self.h)
        rand_vec.requires_grad_(requires_grad=False)
        rand_vec = rand_vec.to(self.learning_class.device)
        
        p_batch = torch.matmul(self.H,p_pred).expand(N, self.h) + torch.diagonal(torch.matmul(self.H,torch.matmul(s_pred, torch.t(self.H)))).sqrt().expand(N, self.h) * rand_vec
        # p_batch size 50 x 6
        p_batch_in_norm1 = ( p_batch - self.dronegate_minimum_2D) / self.dronegate_bounds_2D
        # p_batch_in_norm1 size 50 x 6
        p_batch_in_norm2  = torch.reshape(p_batch_in_norm1, (N, self.h, 1, 1))
        # p_batch_in_norm1 size 50 x 6 x 1 x 1
        if self.baseline == True:
            p_batch_out_norm1, p_batch_out_Rnorm = self.learning_class.encoder(self.learning_class.decoder(p_batch_in_norm2))
            p_batch_out_R = torch.t(p_batch_out_Rnorm.view(p_batch_out_Rnorm.size(0),-1))
            p_Rvec = (1.0/N)*torch.sum(p_batch_out_R, dim=1)

        # p_batch_out_resize size 6 x 50
        elif self.baseline == False:
            p_batch_out_norm1 = self.learning_class.encoder(self.learning_class.decoder(p_batch_in_norm2))
            # p_batch_out_norm1 size 50 x 6 x 1 x 1
            p_batch_out_resize = p_batch_out_norm1.view(p_batch_out_norm1.size(0),-1)
            # p_batch_out_resize size 50 x 6
            p_batch_out = p_batch_out_resize * self.dronegate_bounds_2D + self.dronegate_minimum_2D
            # p_batch_out size 50 x 6
            p_diff = p_batch_out - torch.matmul(self.H, p_pred).expand(N, self.h) #p_batch))
            # p_diff size 50 x 6
            p_diff_bmm_1 = torch.reshape(p_diff, (N, self.h, 1))
            p_diff_bmm_2 = torch.reshape(p_diff, (N, 1, self.h))
            slist = torch.matmul(p_diff_bmm_1, p_diff_bmm_2)

        # expectation
        if self.case_num == 1 or self.case_num == 2:              # maximum likelihood observation
            p_meas = torch.matmul(self.H,p_pred).detach()
        elif self.case_num == 3 or self.case_num == 4:            # approximate future measurement
            p_meas = (1.0/N)*torch.sum(p_batch_out, dim=0)

        if self.baseline == True:
            s_meas = torch.diag(p_Rvec)
        elif self.baseline == False:
            s_meas = (1.0/N)*torch.sum(slist, dim=0)

        # measurement
        y = p_meas - torch.matmul(self.H, p_pred)

        # update
        K = torch.matmul(s_pred, torch.matmul(torch.t(self.H), torch.inverse(torch.matmul(self.H, torch.matmul(s_pred, torch.t(self.H))) + s_meas)))
        p_updt = p_pred + torch.matmul(K, y)
        s_updt = torch.matmul((self.Ic - torch.matmul(K, self.H)), s_pred)

        # normalize all poses
        p_updt_pose_n = (p_updt[0:6] - self.dronegate_minimum) / self.dronegate_bounds
        p_updt_n = torch.cat((p_updt_pose_n, p_updt[6::]))

        p_pred_pose_n = (p_pred[0:6] - self.dronegate_minimum) / self.dronegate_bounds
        p_pred_n = torch.cat((p_pred_pose_n, p_pred[6::]))

        p_meas_n = (p_meas - self.dronegate_minimum) / self.dronegate_bounds
        # print("s_updt: ", s_updt)
        return p_updt_n, p_pred_n, p_meas_n, s_updt, s_pred, s_meas

    def cost_fn(self, opt_iter, p0, s0, u, init=False):
        p_now = p0.clone()
        s_now = s0.clone()
        
        p_updt_traj = p0.clone()
        p_pred_traj = p0.clone()
        p_meas_traj = torch.matmul(self.H,p0)

        s_updt_traj = s0.clone()
        s_pred_traj = s0.clone()
        s_meas_traj = torch.matmul(self.H, torch.matmul(s0, torch.t(self.H)))

        # control limits
        u = torch.clamp(u, self.u_min, self.u_max)

        u = torch.t(u).flatten()
        for i in range(self.k-1):
            # print("k: ", i)
            u_now =  u[i*self.m:i*self.m+self.m]
            u_now.retain_grad()

            p_updt, p_pred, p_meas, s_updt, s_pred, s_meas = self.update_step_tensor(i, p_now, s_now, u_now, self.dt)

            p_updt_traj = torch.cat((p_updt_traj,p_updt),0) # appending to a 1D tensor
            p_pred_traj = torch.cat((p_pred_traj,p_pred),0) # appending to a 1D tensor
            p_meas_traj = torch.cat((p_meas_traj,p_meas),0) # appending to a 1D tensor

            s_updt_traj = torch.cat((s_updt_traj,s_updt),1)
            s_pred_traj = torch.cat((s_pred_traj,s_pred),1)
            s_meas_traj = torch.cat((s_meas_traj,s_meas),1)

            p_now = p_updt.clone()
            p_now.retain_grad()

            s_now = s_updt.clone()
            s_now.retain_grad()

        cost1 = (1./2.)*(1./(self.pconst))*torch.matmul((p_updt_traj-self.xf_tensor),torch.matmul(self.Q_tensor,(p_updt_traj-self.xf_tensor))) 
        cost2 =  (1/(self.k-1))*(1./2.)*torch.matmul(u,torch.matmul(self.R_tensor,u)) # torch.tensor([0.0]).cuda() #
        cost3 = (1./self.varmax)*(1./2.)*torch.trace(s_updt)
        cost = cost1 + cost2 + cost3

        self.cost1list.append(cost1.detach().cpu().numpy())
        self.cost2list.append(cost2.detach().cpu().numpy())
        self.cost3list.append(cost3.detach().cpu().numpy())
        self.costlist.append(cost.detach().cpu().numpy())
        return cost, p_updt_traj, p_pred_traj, p_meas_traj, s_updt_traj, s_pred_traj, s_meas_traj

    def cost_fn_grad(self, opt_iter, p0, s0, u, init = False):
        cost, p_updt_traj, p_pred_traj, p_meas_traj, s_updt_traj, s_pred_traj, s_meas_traj = self.cost_fn(opt_iter, p0, s0, u, init)
        cost.backward()
        cost_gradu = u.grad
        return cost, cost_gradu, p_updt_traj, p_pred_traj, p_meas_traj, s_updt_traj, s_pred_traj, s_meas_traj 

    def loginfo(self, p_updt_numpy, p_pred_numpy, p_meas_numpy, s_updt_numpy, s_pred_numpy, s_meas_numpy, u, i):
        np.savetxt("./empty_save_"+str(self.uncertainty_type)+"/p_updt_numpy_"+str(i)+".txt",p_updt_numpy)
        np.savetxt("./empty_save_"+str(self.uncertainty_type)+"/p_pred_numpy_"+str(i)+".txt",p_pred_numpy)
        np.savetxt("./empty_save_"+str(self.uncertainty_type)+"/p_meas_numpy_"+str(i)+".txt",p_meas_numpy)
        np.savetxt("./empty_save_"+str(self.uncertainty_type)+"/s_updt_numpy_"+str(i)+".txt",s_updt_numpy)
        np.savetxt("./empty_save_"+str(self.uncertainty_type)+"/s_pred_numpy_"+str(i)+".txt",s_pred_numpy)
        np.savetxt("./empty_save_"+str(self.uncertainty_type)+"/s_meas_numpy_"+str(i)+".txt",s_meas_numpy)
        np.savetxt("./empty_save_"+str(self.uncertainty_type)+"/u_"+str(i)+".txt",u.detach().cpu().numpy())
        np.savetxt("./empty_save_"+str(self.uncertainty_type)+"/cost1list_" + str(i) + ".txt", self.cost1list)
        np.savetxt("./empty_save_"+str(self.uncertainty_type)+"/cost2list_" + str(i) + ".txt", self.cost2list)
        np.savetxt("./empty_save_"+str(self.uncertainty_type)+"/cost3list_" + str(i) + ".txt", self.cost3list)
        np.savetxt("./empty_save_"+str(self.uncertainty_type)+"/costlist_" + str(i) + ".txt", self.costlist)
        torch.save(self.optimizer.state_dict(), "./empty_save_"+str(self.uncertainty_type)+"/opt_state_dict.pth")

    def solve_opt(self, mpc_iter, u_init, opt_state_dict, init = False):
        # initialize solution
        u = self.numpy2torch(u_init)
        u = Variable(u, requires_grad=True)
        u_len = u_init.shape[1]
        
        self.optimizer = optim.SGD([u], lr=self.alpha, momentum=0.9)

        if opt_state_dict != None:
            self.optimizer.load_state_dict(torch.load(opt_state_dict))
            
        p0 = self.p0
        p0_tensor = self.numpy2torch(p0)

        s0 = self.s0
        s0_tensor = self.numpy2torch(s0)

        cost = self.numpy2torch(np.array([10.]))

        if self.uncertainty_type == "epistemic":
            var_map = np.loadtxt("./images/varmaps/dronegate/large96k_test_x_map_inv.txt")
        elif self.uncertainty_type == "aleatoric":
            var_map = np.loadtxt("./images/varmaps/dronegate/aleatoric_test_x_map_inv_noisier_checkpt.txt")
        i = 0
        time_array = []
        while cost.item() >= self.max_cost:  
            updt_ellipses = []

            cost, cost_gradu, p_updt_traj, p_pred_traj, p_meas_traj, s_updt_traj, s_pred_traj, s_meas_traj = self.cost_fn_grad(i, p0_tensor, s0_tensor, u, init)
            
            p_updt_numpy = np.reshape(p_updt_traj.detach().cpu().numpy(), (self.n, self.k), order='F')
            p_pred_numpy = np.reshape(p_pred_traj.detach().cpu().numpy(), (self.n, self.k), order='F')
            p_meas_numpy = np.reshape(p_meas_traj.detach().cpu().numpy(), (self.h, self.k), order='F')

            s_updt_numpy = s_updt_traj.detach().cpu().numpy()
            s_pred_numpy = s_pred_traj.detach().cpu().numpy()
            s_meas_numpy = s_meas_traj.detach().cpu().numpy()

            self.optimizer.step()
            self.optimizer.zero_grad()
 
            i += 1

            if i >= self.max_iters:
                self.max_cost = 1.e-4
                break

        
                
        return u, p_updt_numpy, updt_ellipses, i, self.costlist[-1]

# x0 = np.array([0.4, 0.6, 0.5,  0.5, 0.5, 0.5, 0., 0., 0.])
# xf = np.array([0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 0., -1.0, 0.])

# s0 = 0.0001*np.identity(9)
# dt = 0.02
# traj_length = 200
# trajopt_class = TrajectoryOptimization(dt, baseline=False, uncertainty_type="epistemic")
# trajopt_class.logging = False
# trajopt_class.init_traj(x0, s0, xf, traj_length)

# x0_un = copy.deepcopy(x0)
# x0_un[0:6] = (x0[0:6] * trajopt_class.dronegate_bounds.detach().cpu().numpy()) + trajopt_class.dronegate_minimum.detach().cpu().numpy()
# #xf = np.array([0.6, 0.2, 0.5, 0.5, 0.5, 0.5, 0., -1.0, 0.])
# xf_un = copy.deepcopy(xf)
# xf_un[0:6] = (xf[0:6] * trajopt_class.dronegate_bounds.detach().cpu().numpy()) + trajopt_class.dronegate_minimum.detach().cpu().numpy()

 
# ipopt_u, ipopt_x = ipopt_traj(x0_un, xf_un, traj_length)
# u_init = np.zeros((trajopt_class.m, traj_length-1))
# u_init[:,:] = ipopt_u[:,:] 

# trajopt_class.solve_opt(0,u_init,None)
