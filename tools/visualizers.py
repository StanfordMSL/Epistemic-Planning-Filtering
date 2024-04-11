import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import matplotlib.patches as patches 

def plottraj(plan_mode, Xs, Ss, x0, xf, count, uncertainty_mode="epistemic", itr=True):   
    s=5.991   # p = 0.95
    # Initialize Plot
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.set_xlim((-5, 5))
    ax.set_ylim((0, 10))

    if uncertainty_mode == "epistemic":
        cmap = "viridis_r"
    elif uncertainty_mode == "aleatoric":
        cmap = "hot_r"
    var_map = np.loadtxt("tools/" + uncertainty_mode + "_map.txt")

    ax.scatter(var_map[:,0], var_map[:,1], vmin=0, vmax=2, s=10, c=var_map[:,-1], marker='o',cmap=cmap,alpha=0.1)

    dronegate_minimum = np.array([-5., 0., -2.5, -np.pi/2., -np.pi/2., -np.pi, 0., 0., 0.])
    dronegate_bounds  = np.array([10., 10., 5., np.pi, np.pi, np.pi, 0., 0., 0])
    x0 = x0 * dronegate_bounds + dronegate_minimum
    xf = xf * dronegate_bounds + dronegate_minimum

    ax.plot(xf[0],xf[1], "r*" )
    traj_colors = ["cyan", "magenta", "yellow","gray"]

    idx = 0
    for X in Xs:
        if Ss != None:
            S = Ss[idx]
            plot_ellipses = True
        else:
            plot_ellipses = False
        pos = np.array([])
        
        # Plot the Full Trajectory (line)
        ax.plot(X[0,:]*dronegate_bounds[0] + dronegate_minimum[0], X[1,:]*dronegate_bounds[1] + dronegate_minimum[1], c=traj_colors[idx], alpha=1.0)
        ax.scatter(X[0,:]*dronegate_bounds[0] + dronegate_minimum[0], X[1,:]*dronegate_bounds[1] + dronegate_minimum[1],s=1, c=traj_colors[idx], alpha=1.0)

        for i in range(X.shape[1]):
            pos   = (X[0:3, i]*dronegate_bounds[0:3]) + dronegate_minimum[0:3]
            
            if plot_ellipses == True:
                st = S[0:2, i*9:i*9+2]
                l, v = np.linalg.eig(s*st)
                l = np.sqrt(l)
                patch = patches.Ellipse((pos[0], pos[1]), 2*l[0], 2*l[1], np.rad2deg(np.arccos(v[0,0])), edgecolor='c',alpha=0.5, fill=False)
                ax.add_patch(patch)
        idx += 1
    
    if itr == True:
        plt.savefig("planning/plots/" + plan_mode + "_" + str(uncertainty_mode)+ "_" + str(count)+ ".png")
        plt.close()
    else:
        plt.show()

def plotcost(cost1list, cost2list, cost3list, costlist, uncertainty_mode):
    ax  = plt.axes()
    line1, = ax.plot(np.linspace(0,len(costlist),len(costlist)), costlist)
    line2, = ax.plot(np.linspace(0,len(costlist), len(costlist)), cost1list)
    line3, = ax.plot(np.linspace(0,len(costlist), len(costlist)), cost2list)
    line4, = ax.plot(np.linspace(0,len(costlist), len(costlist)), cost3list)
    ax.legend([line1, line2, line3, line4], ["total cost", "state cost", "control cost", "uncertainty cost"])
    ax.set_ylabel("cost")
    ax.set_xlabel("iteration")
    plt.savefig("planning/plots/cost_" + str(uncertainty_mode) + ".png")

    plt.close()

def plotmeasurements(traj_gt, traj_est, traj_z, traj_Q, traj_S, traj_R, gate_count):

    dronegate_minimum = np.array([-5., 0., -2.5, -np.pi/2., -np.pi/2., -np.pi, 0., 0., 0.])
    dronegate_bounds  = np.array([10., 10., 5., np.pi, np.pi, np.pi, 0., 0., 0])

    fig1, axs1 = plt.subplots(3) # position 
    fig2, axs2 = plt.subplots(3) # orientation
    fig3, axs3 = plt.subplots(3) # velocity
    axs_list = [axs1, axs2, axs3]
    num_steps = traj_gt.shape[1]

    epoch_array = np.linspace(0,num_steps,num_steps)
    for i in range(len(axs_list)):
        axs = axs_list[i]
        for j in range(3):
            if i < 2:

                markers_gt, caps_gt, bars_gt    = axs[j].errorbar(epoch_array, traj_gt[i*3+j,:] * dronegate_bounds[i*3+j] + dronegate_minimum[i*3+j], np.sqrt(traj_Q[i*3+j,:]), capsize=5)
                markers_est, caps_est, bars_est = axs[j].errorbar(epoch_array, traj_est[i*3+j,:] * dronegate_bounds[i*3+j] + dronegate_minimum[i*3+j], np.sqrt(traj_S[i*3+j,:]), capsize=5)
                markers_z, caps_z, bars_z       = axs[j].errorbar(epoch_array[0:-1], traj_z[i*3+j,:], np.sqrt(traj_R[i*3+j,:]), capsize=5)
                bars_caps = []
                bars_caps.extend(list(bars_gt))
                bars_caps.extend(list(bars_est))
                bars_caps.extend(list(bars_z))
                bars_caps.extend(list(caps_gt))
                bars_caps.extend(list(caps_est))
                bars_caps.extend(list(caps_z))
                [bar_cap.set_alpha(0.4) for bar_cap in bars_caps]
            else:

                markers_gt, caps_gt, bars_gt    = axs[j].errorbar(epoch_array, traj_gt[i*3+j,:], np.sqrt(traj_Q[i*3+j,:]), capsize=5)
                markers_est, caps_est, bars_est = axs[j].errorbar(epoch_array, traj_est[i*3+j,:], np.sqrt(traj_S[i*3+j,:]), capsize=5)
                bars_caps = []
                bars_caps.extend(list(bars_gt))
                bars_caps.extend(list(bars_est))
                bars_caps.extend(list(caps_gt))
                bars_caps.extend(list(caps_est))
                [bar_cap.set_alpha(0.4) for bar_cap in bars_caps]
    
    axs1[0].set(ylabel = 'x')
    axs1[1].set(ylabel = 'y')
    axs1[2].set(ylabel = 'z')
    axs2[0].set(ylabel = 'phi')
    axs2[1].set(ylabel = 'tht')
    axs2[2].set(ylabel = 'psi')
    axs3[0].set(ylabel = 'vx')
    axs3[1].set(ylabel = 'vy')
    axs3[2].set(ylabel = 'vz')

    fig1.savefig("./images/plots/traj_comparison_position_" + str(gate_count) + ".png")
    plt.close(fig1)

    fig2.savefig("./images/plots/traj_comparison_orientation_" + str(gate_count) + ".png")
    plt.close(fig2)

    fig3.savefig("./images/plots/traj_comparison_velocity_" + str(gate_count) + ".png")
    plt.close(fig3)

    # plt.show()

def plotplan(traj_plan, traj_pred, traj_meas, traj_S, traj_Q, traj_R, bsp_iter):

    dronegate_minimum = np.array([-5., 0., -2.5, -np.pi/2., -np.pi/2., -np.pi, 0., 0., 0.])
    dronegate_bounds  = np.array([10., 10., 5., np.pi, np.pi, np.pi, 0., 0., 0])

    fig1, axs1 = plt.subplots(3) # position 
    fig2, axs2 = plt.subplots(3) # orientation
    fig3, axs3 = plt.subplots(3) # velocity
    axs_list = [axs1, axs2, axs3]
    num_steps = traj_plan.shape[1]

    traj_S_vec = np.diag(traj_S[:, 0:9])
    traj_Q_vec = np.diag(traj_Q[:, 0:9])
    traj_R_vec = np.diag(traj_R[:, 0:6])
    for i in range(1,num_steps):
        traj_S_vec = np.column_stack((traj_S_vec,np.diag(traj_S[:, i*9:i*9+9])))
        traj_Q_vec = np.column_stack((traj_Q_vec,np.diag(traj_Q[:, i*9:i*9+9])))
        traj_R_vec = np.column_stack((traj_R_vec,np.diag(traj_R[:, i*6:i*6+6])))

    epoch_array = np.linspace(0,num_steps,num_steps)
    for i in range(len(axs_list)):
        axs = axs_list[i]
        for j in range(3):
            if i < 2: # position and orientation plots
                markers_plan, caps_plan, bars_plan = axs[j].errorbar(epoch_array, traj_plan[i*3+j,:] * dronegate_bounds[i*3+j] + dronegate_minimum[i*3+j], np.sqrt(traj_Q_vec[i*3+j,:]), capsize=5)
                markers_pred, caps_pred, bars_pred = axs[j].errorbar(epoch_array, traj_pred[i*3+j,:] * dronegate_bounds[i*3+j] + dronegate_minimum[i*3+j], np.sqrt(traj_S_vec[i*3+j,:]), capsize=5)
                markers_meas, caps_meas, bars_meas = axs[j].errorbar(epoch_array, traj_meas[i*3+j,:]* dronegate_bounds[i*3+j] + dronegate_minimum[i*3+j], np.sqrt(traj_R_vec[i*3+j,:]), capsize=5)
                bars_caps = []
                bars_caps.extend(list(bars_plan))
                bars_caps.extend(list(bars_pred))
                bars_caps.extend(list(bars_meas))
                bars_caps.extend(list(caps_plan))
                bars_caps.extend(list(caps_pred))
                bars_caps.extend(list(caps_meas))
                [bar_cap.set_alpha(0.4) for bar_cap in bars_caps]
            else: # velocity plots
                markers_plan, caps_plan, bars_plan    = axs[j].errorbar(epoch_array, traj_plan[i*3+j,:], np.sqrt(traj_Q_vec[i*3+j,:]), capsize=5)
                markers_pred, caps_pred, bars_pred = axs[j].errorbar(epoch_array, traj_pred[i*3+j,:], np.sqrt(traj_S_vec[i*3+j,:]), capsize=5)
                bars_caps = []
                bars_caps.extend(list(bars_plan))
                bars_caps.extend(list(bars_meas))
                bars_caps.extend(list(caps_plan))
                bars_caps.extend(list(caps_meas))
                [bar_cap.set_alpha(0.4) for bar_cap in bars_caps]
    
    axs1[0].set(ylabel = 'x')
    axs1[1].set(ylabel = 'y')
    axs1[2].set(ylabel = 'z')
    axs2[0].set(ylabel = 'phi')
    axs2[1].set(ylabel = 'tht')
    axs2[2].set(ylabel = 'psi')
    axs3[0].set(ylabel = 'vx')
    axs3[1].set(ylabel = 'vy')
    axs3[2].set(ylabel = 'vz')
    
    axs1[0].set_ylim((-2, 2))
    axs1[1].set_ylim((0,15))
    axs1[2].set_ylim((-0.75, 0.75))
    axs2[0].set_ylim((-0.15, 0.15))
    axs2[1].set_ylim((-0.3, 0.35))
    axs2[2].set_ylim((-2, -1.25))
    axs3[0].set_ylim((-0.3,0.3))
    axs3[1].set_ylim((-2.5,0))
    axs3[2].set_ylim((-0.3,0.3))

    fig1.savefig("./images/plots/traj_comparison_position_" + str(bsp_iter) + ".png")
    plt.close(fig1)

    fig2.savefig("./images/plots/traj_comparison_orientation_" + str(bsp_iter) + ".png")
    plt.close(fig2)

    fig3.savefig("./images/plots/traj_comparison_velocity_" + str(bsp_iter) + ".png")
    plt.close(fig3)

