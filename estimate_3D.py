import torch
import numpy as np
import utils
import math
import copy
from A_calc import a_calc

class EKF():
    def __init__(self, n, m, h, dt):
        # CONSTANTS
        self.n = n
        self.m = m
        self.h = h
        self.mu = np.zeros(n)
        self.sigma = 0.000001*np.identity(n)
        self.Q = np.identity(self.n) # 0.05*np.identity(n)
        self.Q[0:3, 0:3] = 1.e-5 * self.Q[0:3, 0:3]  # position
        self.Q[3:6, 3:6] = 1.e-6 * self.Q[3:6, 3:6]  # orientation
        self.Q[6::, 6::] = 1.e-6 * self.Q[6::,6::]   # velocity
        print("Q: ", self.Q)
        self.dt = dt
        self.H = np.zeros((self.h, self.n))
        self.H[0:6,0:6] = np.eye(self.h)
        self.dronegate_minimum = np.array([-5., 0., -2.5, -np.pi/2., -np.pi/2., -np.pi])
        self.dronegate_bounds  = np.array([10., 10., 5., np.pi, np.pi, np.pi])

        self.A = None

    def getR(self, learning_class, z):
        N = 50
        mu_phys = self.mu.copy()
        mu_phys[0:6] = (self.mu[0:6] * self.dronegate_bounds) + self.dronegate_minimum

        # sample
        rand_vec = torch.randn(self.h, N).to(learning_class.device)
        mu_tensor = utils.numpy2torch(mu_phys, learning_class)
        sigma_tensor = utils.numpy2torch(self.sigma,learning_class)
        H_tensor = utils.numpy2torch(self.H, learning_class)
        dronegate_bounds_torch = utils.numpy2torch(self.dronegate_bounds, learning_class)
        dronegate_minimum_torch = utils.numpy2torch(self.dronegate_minimum, learning_class)
        z_tensor = utils.numpy2torch(z, learning_class)

        p_batch = torch.t(torch.matmul(H_tensor,mu_tensor).expand(N, self.h)) + torch.t(torch.diagonal(torch.matmul(H_tensor,torch.matmul(sigma_tensor, torch.t(H_tensor)))).sqrt().expand(N, self.h)) * rand_vec
        p_batch_in_norm1 = ( p_batch - torch.reshape(dronegate_minimum_torch, (6,1)) ) / torch.reshape(dronegate_bounds_torch, (6,1))

        p_batch_in_norm2 = torch.reshape(torch.t(p_batch_in_norm1), (N, self.h, 1, 1))
        p_batch_out_norm1 = learning_class.encoder(learning_class.decoder(p_batch_in_norm2))
        if len(p_batch_out_norm1) != 2:
            p_batch_out_resize = torch.t(p_batch_out_norm1.view(p_batch_out_norm1.size(0),-1))
            p_batch_out = p_batch_out_resize * torch.reshape(dronegate_bounds_torch,(6,1)) + torch.reshape(dronegate_minimum_torch, (6,1))
            
            p_diff1 = torch.t(p_batch_out - torch.t(torch.matmul(H_tensor, mu_tensor).expand(N, self.h)))
            p_diff2 = z_tensor.expand(N, self.h) - torch.matmul(H_tensor, mu_tensor).expand(N, self.h)

            p_diff = torch.vstack((p_diff1, p_diff2))
            
            p_diff_bmm_1 = torch.reshape(p_diff, (2*N, self.h, 1))
            p_diff_bmm_2 = torch.reshape(p_diff, (2*N, 1, self.h))

            slist = torch.matmul(p_diff_bmm_1, p_diff_bmm_2)
            R_tensor = (1.0/(2*N))*torch.sum(slist,dim=0)  #(torch.sum(slist1, dim=0) + torch.sum(slist2,dim=0))
            R_numpy = R_tensor.detach().cpu().numpy()
            
            mu_sample = p_batch.detach().cpu().numpy()
            z_sample = p_batch_out.detach().cpu().numpy()
        
        return z_sample, mu_sample, R_numpy

    def quad_dynamics(self, u):
        # unnormalize mu
        mu_phys = self.mu.copy()
        mu_phys[0:6] = (self.mu[0:6] * self.dronegate_bounds) + self.dronegate_minimum

        self.A = a_calc(mu_phys, u, self.dt, m=1.0)
        # define constants
        grav = np.array([0., 0., 9.81])
        zb = np.array([0, 0, 1])

        Rx = np.array([[1, 0, 0],[0, math.cos(mu_phys[3]), -math.sin(mu_phys[3])],[0, math.sin(mu_phys[3]), math.cos(mu_phys[3])]])
        Ry = np.array([[math.cos(mu_phys[4]), 0, math.sin(mu_phys[4])],[0, 1, 0],[-math.sin(mu_phys[4]), 0, math.cos(mu_phys[4])]])
        Rz = np.array([[math.cos(mu_phys[5]), -math.sin(mu_phys[5]), 0],[math.sin(mu_phys[5]), math.cos(mu_phys[5]), 0],[0, 0, 1]])

        body2world_mtx = Rz @ Ry @ Rx
        a = -grav + np.dot(body2world_mtx, u[0]*zb)

        body2euler_rate_mtx = np.array([[1, math.sin(mu_phys[3])*math.tan(mu_phys[4]), math.cos(mu_phys[3])*math.tan(mu_phys[4])],
                                        [0, math.cos(mu_phys[3]), -math.sin(mu_phys[3])], 
                                        [0, math.sin(mu_phys[3])*(1./math.cos(mu_phys[4])), math.cos(mu_phys[3])*(1./math.cos(mu_phys[4]))]])

        v = mu_phys[6:9] + self.dt*a
        p = mu_phys[0:3] + self.dt*mu_phys[6:9]
        th = mu_phys[3:6] + self.dt*(body2euler_rate_mtx @ u[1::])
        x_new = np.hstack((p,th,v))
        
        return x_new

    # extended kalman filter
    def ekf(self, u, z, learning_class, GPS=False):
        z_sample = np.zeros((6,))
        mu_sample = np.zeros((9,))
        
        # predict step
        mu_pred = self.quad_dynamics(u) 
        self.mu = copy.deepcopy(mu_pred)
        self.mu[0:6] = (self.mu[0:6] - self.dronegate_minimum) / self.dronegate_bounds
        self.sigma = self.A @ self.sigma @ self.A.T + self.Q
        # compute R 
        if GPS == False:
            if len(z) == 2:
                R = z[1]
                z = z[0]
            else:
                z_sample, mu_sample, R = self.getR(learning_class,z)
        elif GPS == True:
            if len(z) == 2:
                z = z[0]
            R = np.identity(self.h)
            R[0:3,0:3] = 2.5e-3 * R[0:3,0:3]
            R[3:6,3:6] =  4.e-4 * R[3:6,3:6]
            
        y = z - np.dot(self.H, mu_pred)
        K = np.linalg.multi_dot([self.sigma,self.H.T,np.linalg.inv(np.linalg.multi_dot([self.H,self.sigma,self.H.T]) + R)])
        self.mu = mu_pred + np.dot(K, y)
        self.sigma = np.dot(np.identity(self.n) - K@self.H, self.sigma)
        self.mu[0:6] = (self.mu[0:6] - self.dronegate_minimum) / self.dronegate_bounds
        return z_sample, mu_sample, R

    # Simulate one forward pass of the EKF
    def unit_test(self):
        self.mu = np.random.rand(self.n)
        self.sigma = np.random.rand(self.n, self.n)
        u = np.array([0.01, 0.01, 0.01, 0.01])
        z = np.ones(self.h)
        R = np.identity(self.h)
        self.ekf(u, z, None)

def main():#n, m, h, dt
    ekf_class = EKF(9, 4, 6, 0.01)
    ekf_class.unit_test()
    print(ekf_class.mu, ekf_class.sigma)

if __name__ == "__main__":
    main()
