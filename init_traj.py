import casadi
import numpy as np
import copy
from visualizers import trajviz3D
def ipopt_traj(x0, xf, k, x_init=np.array([None]), u_init=np.array([None]), baseline=False):
    cost = 2000.
    iter = 0
    while cost > 1000.:
        dronegate_minimum = np.array([-5., 0., -2.5, -np.pi/2., -np.pi/2., -np.pi])
        dronegate_bounds  = np.array([10., 10., 5., np.pi, np.pi, np.pi])
        if u_init.any() == None:
            u0 = np.array([9.81, 0.0, 0.0, 0.0])
            x0_in = np.tile(x0, k)
            u0_in = np.tile(u0, k-1)
        else:
            x0_in = (x_init[0:6,:] * dronegate_bounds.reshape((-1,1))) + dronegate_minimum.reshape((-1,1))
            x0_in = np.vstack((x0_in, x_init[6::, :]))
            x0_in[:,0] = x0
            x0_in = x0_in.T.flatten()
            u0_in = u_init
        if baseline == False: 
            pconst = 1.0
            CQ = casadi.SX(np.identity(9))
            CR = casadi.SX(0.1*np.identity(4*(k-1)))
        elif baseline == True:
            pconst = 1.e-5
            CQ = casadi.SX(np.identity(9))
            CQ[3:6, 3:6] = 0.1*CQ[3:6, 3:6]
            CQ[6::, 6::] = 0.0*CQ[6::,6::]
            a = np.array([0.1, 1., 1., 1.]) # thrust roll pitch yaw
            b = np.tile(a,k-1)
            CR = casadi.SX(np.diag(b))

        m = 1.0
        grav = casadi.SX(np.array([0.0, 0.0, 9.81]))
        dt = 0.02


        u_min = np.array([0.0,   -3.5, -3.5, -2.0])
        u_max = np.array([9.81*2, 3.5,  3.5,  2.0])

        # optimization variables
        x = casadi.SX.sym('x',9*k)
        u = casadi.SX.sym('u',4*(k-1))
        g = [x[0:9] - x0]
        # dynamic constraints
        x_new = casadi.SX.zeros(9*k)
        x_new[0:9] = x0

        for i in range(k-1):
            # dynamics for a single time step
            xt = x[i*9:i*9+9]
            ut = u[i*4:i*4+4]

            # Unpack States
            # position    = xt[0:3]
            orientation = xt[3:6]
            velocity    = xt[6:9]
            
            # Unpack Inputs
            f_in    = casadi.SX.zeros(3,)
            f_in[2] = ut[0]
            br_in   = ut[1:4]

            # Useful Intermediate Terms
            Rx1 = casadi.SX(np.array([[1, 0, 0],[0, 0, 0],[0, 0, 0]]))
            Rx2 = casadi.sin(orientation[0])*casadi.SX(np.array([[0, 0, 0],[0, 0,-1],[0, 1, 0]]))
            Rx3 = casadi.cos(orientation[0])*casadi.SX(np.array([[0, 0, 0],[0, 1, 0],[0, 0, 1]]))
            Rx = Rx1 + Rx2 + Rx3

            Ry1 = casadi.SX(np.array([[0, 0, 0],[0, 1, 0],[0, 0, 0]]))
            Ry2 = casadi.sin(orientation[1])*casadi.SX(np.array([[0, 0, 1],[0, 0, 0],[-1, 0, 0]]))
            Ry3 = casadi.cos(orientation[1])*casadi.SX(np.array([[1, 0, 0],[0, 0, 0],[ 0, 0, 1]]))
            Ry = Ry1 + Ry2 + Ry3

            Rz1 = casadi.SX(np.array([[0, 0, 0],[0, 0, 0],[0, 0, 1]]))
            Rz2 = casadi.sin(orientation[2])*casadi.SX(np.array([[0,-1, 0],[1, 0, 0],[0, 0, 0]]))
            Rz3 = casadi.cos(orientation[2])*casadi.SX(np.array([[1, 0, 0],[0, 1, 0],[0, 0, 0]]))
            Rz = Rz1 + Rz2 + Rz3
            
            R = casadi.mtimes(Rz, casadi.mtimes(Ry,Rx))

            body2euler_rate1 = casadi.SX(np.array([[1, 0, 0],[0, 0, 0],[0, 0, 0]]))
            body2euler_rate2 = casadi.sin(orientation[0])*casadi.tan(orientation[1])*casadi.SX(np.array([[0, 1, 0],[0, 0, 0],[0, 0, 0]]))
            body2euler_rate3 = casadi.cos(orientation[0])*casadi.tan(orientation[1])*casadi.SX(np.array([[0, 0, 1],[0, 0, 0],[0, 0, 0]]))
            body2euler_rate4 = casadi.cos(orientation[0])*casadi.SX(np.array([[0, 0, 0],[0, 1, 0],[0, 0, 0]]))
            body2euler_rate5 = casadi.sin(orientation[0])*casadi.SX(np.array([[0, 0, 0],[0, 0,-1],[0, 0, 0]]))
            body2euler_rate6 = casadi.sin(orientation[0])*(1./casadi.cos(orientation[1]))*casadi.SX(np.array([[0, 0, 0],[0, 0, 0],[0, 1, 0]]))
            body2euler_rate7 = casadi.cos(orientation[0])*(1./casadi.cos(orientation[1]))*casadi.SX(np.array([[0, 0, 0],[0, 0, 0],[0, 0, 1]]))
            T = body2euler_rate1 + body2euler_rate2 + body2euler_rate3 + body2euler_rate4 + body2euler_rate5 + body2euler_rate6 + body2euler_rate7

            F_g = -m*grav
            F_t = casadi.mtimes(R, f_in)

            p_dot = velocity

            v_dot = (F_g + F_t)
            o_dot = casadi.mtimes(T, br_in)
            # print("dynamics done")

            xdot = casadi.SX.zeros(9)
            xdot[0:3] = p_dot
            xdot[3:6] = o_dot
            xdot[6:9] = v_dot
            x_upd = xt + dt*xdot

            x_new[(i+1)*9:(i+1)*9+9] = x_upd

            g.append(x_upd - x[(i+1)*9:(i+1)*9+9])


        cost = 0.5*(1./pconst)*casadi.mtimes(casadi.transpose((x[(i+1)*9:(i+1)*9+9]-xf)),casadi.mtimes(CQ,(x[(i+1)*9:(i+1)*9+9]-xf))) + 0.5*(1./(k-1))*casadi.mtimes(casadi.transpose(u),casadi.mtimes(CR,u))
        # define the cost function
        nlp = {}
        nlp['x'] = casadi.vertcat(x,u)
        nlp['f'] = cost
        nlp['g'] = casadi.vertcat(*g)
        opts = {'ipopt.print_level':0, 'print_time':0}
        F = casadi.nlpsol('F','ipopt',nlp, opts)

        ubx = np.hstack((np.inf * np.ones((9*k,)), np.tile(u_max, k-1)))
        lbx = np.hstack((-np.inf * np.ones((9*k,)), np.tile(u_min, k-1)))
        res = F(x0=np.hstack((x0_in.reshape((1,-1)), u0_in.reshape((1,-1)))), ubx=ubx, lbx=lbx, ubg=0, lbg=0)

        x_flat = res['x'][0:9*k]
        cost = res['f']
        x_traj = x_flat.reshape((9, k))
        x_norm = (x_traj[0:6, :] - dronegate_minimum) / dronegate_bounds
        x_norm = np.vstack((x_norm, x_traj[6::,:]))

        u_flat = res['x'][9*k::]
        u_traj = u_flat.reshape((4, k-1))

        u_init = np.zeros((4, k-1))
        x_init = np.zeros((9, k))

        u_init[:,:] = u_traj[:,:] 
        x_init[:,:] = x_norm[:,:]

        iter += 1

    return u_traj, x_norm


