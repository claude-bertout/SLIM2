"""
module name: plot_results. Functions in this module are called by main after the end of the computation
to produce publication quality plots from the model computation data. Plots are stored in a
subdirectory of results identified by a date-time stamp of the computation start time.
"""

import numpy as np
import matplotlib.pyplot as plt
import import_parameters as prm
import set_grids as grid
import function_library as lib
import in_out_library as io
import constants as const
import math

# define various quantities needed for the plots
if prm.velocity_index == 1 or prm.velocity_index == 3:
    line_color = ['r', 'g', 'b']
    wave_1 = ['R', 'C', 'B']
    wave_2 = ['H $\lambda$3970', 'FeI $\lambda$3969', 'CaII $\lambda$3968']
    wav_limits = [3965, 3976]
else:
    line_color = ['b', 'g', 'r']
    wave_1 = ['B', 'C', 'R']
    wave_2 = ['CaII $\lambda$3968', 'FeI $\lambda$3969', 'H $\lambda$3970']
    wav_limits = [3965, 3974]
lab_1 = ['{}'.format(wave_1[0]), '{}'.format(wave_1[1]), '{}'.format(wave_1[2])]
lab_2 = ['{}'.format(wave_2[0]), '{}'.format(wave_2[1]), '{}'.format(wave_2[2])]

cm = 1 / 2.54  # centimeters in inches


# ----------------------------------------
def plot_absorption_profiles(x_axis, dxd):
    # ------------------------------------
    fig, ax = plt.subplots()
    ax.set_title('Absorption profiles')

    for lx in range(prm.nline):
        plt.plot(x_axis, dxd[lx, :], color=line_color[lx], label=lab_2[lx])
    ax.set_xlabel('Arbitrary frequency scale')
    ax.set_ylabel('Local absorption profile intensity')
    ax.legend(loc='best')
    plt.show()


# ----------------------------
def plot_l_source(source, fc):
    # ------------------------
    """
    Called in source_module compute_local_source_function
    """
    fig, ax = plt.subplots()
    ax.set_title('Normalized local line source functions')
    for lx in range(prm.nline):
        plt.plot(grid.r[:], np.log10(source[lx, :] / fc[lx]), '-', color=line_color[lx], label=lab_2[lx])
    ax.set_xlabel('r/Rc')
    ax.set_ylabel('Log local source function')
    ax.legend(prop={'size': 8}, loc='best')
    plt.show()


# ----------------------------
def plot_nl_source(source_nl):
    # ------------------------
    """
    plots the final source functions for the 3 lines
    """
    fig, ax = plt.subplots()
    # fig.suptitle('Line source functions', fontsize='medium')
    # ax.set_xlim(0.95, prm.rmax * 1.05)
    # ax.set_ylim(-3.5, 0.0)
    ax.set_box_aspect(1)

    for lx in range(prm.nline):
        ax.plot(grid.r, np.log10(source_nl[lx, :]), '-', color=line_color[lx], label=lab_2[lx])

    ax.set_xlabel('$r/R_c$', fontsize='small')
    ax.set_ylabel('$\log (S_\lambda / I_c)$', fontsize='small')
    ax.tick_params(labelcolor='b', labelsize='8', width=1)
    ax.tick_params(labelcolor='b', labelsize='8', width=1)
    ax.legend(prop={'size': 8}, loc='upper right')


# -----------------------------------
def plot_sources(source_l, source_nl):
    # -------------------------------
    """
    Called in main. plots the line source functions, normalized to the local continuum flux, as a function of radius
    """
    fig, ax1 = plt.subplots()
    ax1.set_title('Normalized line source functions')
    ax2 = ax1.twinx()
    ax1.set_ylim(-2.5, 1.0)
    ax2.set_ylim(-2.5, 1.0)
    for lx in range(prm.nline):
        ax1.plot(grid.r[:], np.log10(source_l[lx, :]), dashes=[2, 2], color=line_color[lx], label=lab_2[lx])
        ax2.plot(grid.r[:], np.log10(source_nl[lx, :]), '-', color=line_color[lx], label=lab_2[lx])
    ax1.set_xlabel('r/R$_c$')
    ax1.set_ylabel('Log local source functions (dashed lines)')
    ax2.set_ylabel('Log final source functions (solid lines)')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    fig.tight_layout()

    plt.show()


# -------------------------
def plot_beta(beta1, beta2, str1, str2):
    # ---------------------
    """
    Called in source_module rg2rl: plots the escape probability functions beta and betac that appear
    in the local source function expressions
    """
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_title(f"Log {str1} and {str2} functions")
    for lx in range(prm.nline):
        ax1.plot(grid.r, np.log10(beta1[lx, :]), '-', color=line_color[lx], label=lab_2[lx])
        ax2.plot(grid.r, np.log10(beta2[lx, :]), dashes=[2, 2], color=line_color[lx], label=lab_2[lx])
    ax1.set_ylim(-2.5, 1.0)
    ax2.set_ylim(-4.5, -1.0)
    ax1.set_xlabel('r/Rc')
    ax1.set_ylabel(f"Log {str1} (solid line)")
    ax2.set_ylabel(f"Log {str2} (dashed line)")
    ax1.tick_params(axis='y', labelcolor='red')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax1.legend(prop={'size': 8}, loc='best')
    ax2.legend(prop={'size': 8}, loc='best')
    fig.tight_layout()
    plt.show()


# ----------------------------------------
def plot_rl(igauss, rplot, rsymb, xl, yl):
    # ------------------------------------
    """
    Called in source_module rad_vel_over_space: plots the set of rays over which the source gaussian integrations
    are performed at radius r[rplot]
    """
    fig, ax = plt.subplots()
    plt.xlim(-grid.r[-1] * 1.05, grid.r[-1] * 1.05)
    plt.ylim(-0.05, grid.r[-1] * 1.05)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_xlabel('x$_l$/R$_c$')
    ax.set_ylabel('y$_l$/R$_c$')
    ax.set_title(f"r$_l$ coordinates for r = {grid.r[rplot]:.2f}")
    for j in range(igauss):
        ax.plot(xl[rplot, j], yl[rplot, j], rsymb)
    plt.show()


# ------------------------------------------------
def plot_vrad(igauss, rplot, xl, yl, vrad_on_ray):
    # --------------------------------------------
    """
    Called in rad_vel_over_space. Plots the radial velocities in all directions at r[rplot]
    """
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x$_l$ coordinate')
    ax.set_ylabel('y$_l$ coordinate')
    ax.set_zlabel('vrad_on_ray')
    ax.set_title(f"vrad_on_ray for r = {grid.r[rplot]:.2f}")
    for j in range(igauss):
        ax.plot3D(xl[rplot, j], yl[rplot, j], vrad_on_ray[rplot, j])
    plt.show()
    
    
# ---------------------------
def plot_r0(igauss, lx, lxx, x0, y0):
    # -----------------------
    """
    Called in locate_cp_surfaces. Plotting function for the CP surfaces r0 in cartesian coordinates
    """
    fig, ax = plt.subplots()
    thetac = [np.deg2rad(t) for t in range(0, 91)]
    xc = np.cos(thetac)
    yc = np.sin(thetac)
    xm = grid.r[-1] * np.cos(thetac)
    ym = grid.r[-1] * np.sin(thetac)
    for rplot, rsymb in grid.rplots.items():
        plt.xlim(-0.05, prm.rmax * 1.05)
        plt.ylim(-0.05, prm.rmax * 1.05)
        plt.gca().set_aspect('equal', adjustable='box')
        ax.set_xlabel('x$_0$')
        ax.set_ylabel('y$_0$')
        ax.set_title(f"CP surface for lx = {lx}, lxx = {lxx}")
        plt.plot(xc, yc, '-')
        plt.plot(grid.r[rplot], 0, '*', color=rsymb)
        plt.plot(xm, ym, '-')
        for j in range(igauss):
            if x0[lx, lxx, rplot, j] and y0[lx, lxx, rplot, j]:
                plt.plot(x0[lx, lxx, rplot, j], y0[lx, lxx, rplot, j], '.', color=rsymb)
    plt.show()


# ----------------
def plot_eta(eta):
    # ------------
    """
    called in source_module eta_interact_integral.
    """
    fig, ax = plt.subplots()
    ax.set_title('non-local source function $\eta$ integral')
    for lx in range(prm.nline):
        if np.any(eta[lx, :]):
            plt.plot(grid.r[:], np.log10(eta[lx, :]), '-', color=line_color[lx], label=lab_2[lx])
    ax.set_xlabel('r/R$_c$')
    ax.set_ylabel('log $\eta$ integral')
    ax.tick_params(labelcolor='b', labelsize='8', width=1)
    ax.tick_params(labelcolor='b', labelsize='8', width=1)
    ax.legend(prop={'size': 8}, loc='upper right')
    plt.show()


# -----------------------------
def plot_3x3_crvs(delta_v_mat):
    # -------------------------
    """
    This subroutine plots the constant radial velocity surfaces, as seen by an observer located at x = \infty
    for the 3 interacting lines. The flux for a given line is computed by integrating the emergent intensities
    from the 3 regions shown in the same column of the plot.
    """

    # --------------------------------
    def vrad_z_prime(rd, v_env, p, z):
        # ----------------------------
        """
        vrad_z_prime returns the radial velocity in direction z for all z's at p=cst
        it is the same as lib.vrad_z with a different set of parameters
        valid for any velocity field
        """
        rx = np.sqrt(p ** 2 + z ** 2)
        vx = np.interp(rx, rd, v_env)
        cos_a = z / rx
        return vx * cos_a

    # --------------------------------------------------------------------------------------------
    # prepare plot area (3 columns of 3 graphs each)
    # --------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(3, 3, constrained_layout=True, sharex=True, sharey=True,
                            gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(21 * cm, 16 * cm))

    gridspec = axs[0, 0].get_subplotspec().get_gridspec()
    # fig.suptitle(
    #     r'$\lambda\lambda$' + '{}'.format(wave[0]) + ', ' + '{}'.format(wave[1]) + ', ' + \
    #     '{}'.format(wave[2]) + ' interactions', fontsize='large')

    # clear the left column for the subfigure:
    for a in axs[:, 0]:
        a.remove()

    # make the left subfigure in the empty gridspec slots:
    subfigl = fig.add_subfigure(gridspec[:, 0])
    axsLeft = subfigl.subplots(3, 1, sharey=True)

    # clear the center column for the subfigure:
    for a in axs[:, 1]:
        a.remove()

    # make the center subfigure in the empty gridspec slots:
    subfigl = fig.add_subfigure(gridspec[:, 1])
    axsCenter = subfigl.subplots(3, 1, sharey=True)

    # clear the right column for the subfigure:
    for a in axs[:, 2]:
        a.remove()

    # make the right subfigure in the empty gridspec slots:
    subfigr = fig.add_subfigure(gridspec[:, 2])
    axsRight = subfigr.subplots(3, 1)

    # ------------------------------------------
    # compute and plot crvs surfaces for 3 lines
    # ------------------------------------------
    pc = prm.rc
    pmax = prm.rmax
    prange = 400
    zrange = 400
    pj = np.linspace(0.001 * pc, pmax * 0.999, prange)
    # pj[0] = pj[0] * 1.001
    # pj[-1] = pj[-1] * 0.999
    # print(f"pj\n{pj}")

    # plot parameters
    dv = 100

    vs = int(math.ceil((max(np.abs(grid.v_env)) / dv))) * dv
    # print(f"v_env_max = {max(grid.v_env)}, vs = {vs}")
    vt = [k for k in range(-vs, vs + dv, dv)]
    vs = max(grid.v_env)
    # print(f"vs = {vs}, vt = {vt}")

    # choose cmap
    chmap = 'coolwarm_r'

    # control colorbar size
    reduc = 1

    # data for plotting core and envelope sizes
    thetac = [np.deg2rad(t) for t in range(0, 181)]
    xc = np.cos(thetac)
    yc = np.sin(thetac)
    xm = prm.rmax * np.cos(thetac)
    ym = prm.rmax * np.sin(thetac)

    # define z range for each p
    zmin = np.zeros(zrange)
    zmax = np.sqrt(prm.rmax ** 2 - pj ** 2)
    # print(f"zmax\n{zmax}")
    k = -1
    for pjk in pj:
        k += 1
        if pjk < prm.rc:
            zmin[k] = np.sqrt(prm.rc ** 2 - pjk ** 2)
        else:
            zmin[k] = -np.sqrt(prm.rmax ** 2 - pjk ** 2)

    # print(f"zmin\n{zmin}")

    vradz = np.zeros((prange, zrange, prm.nline, prm.nline))
    vradzp = np.zeros((prange, zrange))
    vradzp_max = np.zeros(prange)
    vradzp_min = np.zeros(prange)
    zpj = np.linspace(-prm.rmax, prm.rmax, zrange)
    # print(f"zpj \n{zpj}")

    # ------------------------------------
    # plot undisplaced and displaced CRVSs
    # ------------------------------------
    for i in range(prange):
        vradzp[i, :] = vrad_z_prime(grid.r, grid.v_env, pj[i], zpj)
        # if i == 0:
        #     print(f"vradzp[0, :] \n {vradzp[i, :]}")
        vradzp_max[i] = np.max(vradzp[i])
        vradzp_min[i] = np.min(vradzp[i])
        # print(f"vradzp_min[{i}]= {vradzp_min[i]}, vradzp_max[{i}] = {vradzp_max[i]}")

        for j in range(zrange):
            for lx in range(prm.nline):
                for mx in range(prm.nline):
                    if zmin[i] < zpj[j] < zmax[i]:
                        vradz[i, j, lx, mx] = vrad_z_prime(grid.r, grid.v_env, pj[i], zpj[j]) - delta_v_mat[lx, mx]
                    else:
                        vradz[i, j, lx, mx] = None
                    # keep only frequencies present on the ray
                    if vradzp_min[i] < vradz[i, j, lx, mx] < vradzp_max[i]:
                        pass
                    else:
                        vradz[i, j, lx, mx] = None

    # In order to handle accelerating infall or non-monotonic infall, use
    # negative vradz rather than negative velocity field.
    # This is equivalent to reversing the wavelengths of computed profiles.

    if prm.velocity_index == 1 or prm.velocity_index == 3:
        vradz = - vradz
    mx = -3
    for lx in range(prm.nline):
        for grid_position in [axsLeft, axsCenter, axsRight]:
            if grid_position.any() == axsLeft.any():
                mx = 0
            if grid_position.any() == axsCenter.any():
                mx = 1
            if grid_position.any() == axsRight.any():
                mx = 2
            grid_position[lx].set_xlim(-prm.rmax, prm.rmax)
            grid_position[lx].set_ylim(0.0, prm.rmax)
            grid_position[lx].set_box_aspect(0.5)

            # grid_position[lx].tick_params(labelcolor='b', labelsize='small', width=1,
            # labelleft=False, labelbottom=False)
            # remove the x and y ticks
            grid_position[lx].set_xticks([])
            grid_position[lx].set_yticks([])

            xf = [x for x in np.linspace(-prm.rmax, prm.rmax, 100)]
            yf = [0 for _ in np.linspace(-prm.rmax, prm.rmax, 100)]
            grid_position[lx].plot(xf, yf, '-', linewidth=2, color='grey')
            grid_position[lx].plot(xc, yc, '-', linewidth=1, color='grey')
            grid_position[lx].plot(xm, ym, '-', linewidth=1, color='grey')
            # remove frames
            grid_position[lx].set_frame_on(False)

            if lx == mx:
                grid_position[lx].set_title('{}'.format(wave_1[lx]) + '-line CRVSs (km/s)', fontsize='medium')
            else:
                grid_position[lx].set_title('{}'.format(wave_1[lx]) + '-line contribution to line ' +
                                            '{}'.format(wave_1[mx]), fontsize='medium')

            pc0 = grid_position[lx].contourf(zpj, pj, vradz[:, :, lx, mx], 40, cmap=chmap, vmin=-vs, vmax=vs)
            cbar0 = subfigl.colorbar(pc0, shrink=reduc, ax=grid_position[lx], location='bottom', ticks=vt)
            cbar0.ax.tick_params(labelsize=8)


# ------------------
def plot_v_and_nt():
    # --------------
    fig, ax1 = plt.subplots()
    # fig.suptitle('Stellar wind properties', fontsize='medium')
    if prm.velocity_index == 1 or prm.velocity_index == 3:
        ax1.plot(grid.r, -grid.v_env, color='red')
    else:
        ax1.plot(grid.r, grid.v_env, color='red')

    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_xlabel('$r/R_c$')
    ax1.set_ylabel('Velocity $v(r)$ (km/s)')

    ax2 = ax1.twinx()
    ax2.plot(grid.r, np.log10(grid.nt), color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    # ax2.set_xlabel('radius')
    ax2.set_ylabel('Log number density $n_t$ (cm$^{-3}$)')
    fig.tight_layout()


# --------------------------------
def plot_opacities(taur, epsilon):
    # ----------------------------
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx() if epsilon.any() else None
    for lx in range(prm.nline):
        ax1.plot(grid.r, np.log10(taur[lx, :]), '-', color=line_color[lx], label=lab_2[lx])
        if epsilon.any():
            ax2.plot(grid.r, np.log10(epsilon[lx, :]), dashes=[6, 2], color=line_color[lx])
    ax1.set_xlabel('r/Rc')
    ax1.set_ylabel('log absorption coefficient (cm$^2$g$^{-1}$)')
    ax2.set_ylabel('log collisional de-excitation rate') if epsilon.any() else None
    ax1.tick_params(axis='y', labelcolor='red')
    ax2.tick_params(axis='y', labelcolor='blue') if epsilon.any() else None
    ax1.legend(prop={'size': 8}, loc='upper right')
    fig.tight_layout()


# ------------------------------------------
def read_and_plot_2_cp_surfaces(in_file_cp):
    # --------------------------------------
    """
    a CP surface plot tailored to the case of 3-line interactions in accelerating outflows,
    in which case the bluest line is entirely local (no cp)
    """
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(18 * cm, 9 * cm))
    # fig.suptitle(r'Common-point surfaces', fontsize='large')

    # read and plot cp and ri surfaces (stored in cartesian coordinates)
    x0 = np.zeros((prm.nline, prm.nline, prm.idr, prm.igauss_shell))
    y0 = np.zeros((prm.nline, prm.nline, prm.idr, prm.igauss_shell))
    thetac = [np.deg2rad(t) for t in range(0, 91)]
    with open(in_file_cp, "r") as f:
        while True:
            one_line = f.readline()
            if not one_line:
                break
            v0, v1, v2, v3, v4, v5, v6 = one_line.split()
            rplot = int(v0)
            color = v1
            igauss = int(v2)
            xplot = float(v3)
            yplot = float(v4)
            lx = int(v5)
            lxx = int(v6)
            # print(f"\nrplot = {rplot}, color = {color}, igauss = {igauss}, xplot = {xplot}, "
            #       f"yplot = {yplot}, lx = {lx}, lxx = {lxx}")

            for j in range(igauss):
                xvalue, yvalue = f.readline().split()
                if float(xvalue) or float(yvalue):
                    x0[lx, lxx, rplot, j] = float(xvalue)
                    y0[lx, lxx, rplot, j] = float(yvalue)
                else:
                    x0[lx, lxx, rplot, j] = None
                    y0[lx, lxx, rplot, j] = None
                # print(x0[lx, lxx, rplot, j], y0[lx, lxx, rplot, j])

            if not np.any(x0[lx, lxx, rplot, :]):
                print('no cp-surface for this combination')
                pass
            else:
                if lx == 1:
                    axs[0].set_title('Selected CP-surfaces for ' + '{}'.format(wave_1[1]), fontsize='small')
                    axs[0].set_xlim(0.0, prm.rmax)
                    axs[0].set_ylim(0.0, prm.rmax)
                    axs[0].set_box_aspect(1.0)
                    axs[0].set_xlabel('$r/R_c$', fontsize='small')
                    axs[0].set_ylabel('$r/R_c$', fontsize='small')
                    axs[0].tick_params(labelcolor='b', labelsize='8', width=1)
                    axs[0].tick_params(labelcolor='b', labelsize='8', width=1)
                    # axs[0].text(7, 9, 'CP-surfaces\nfor $\lambda$3970', color='grey', fontsize=8)
                    xc = np.cos(thetac)
                    yc = np.sin(thetac)
                    axs[0].plot(xc, yc, '-', linewidth=0.5, color='k')
                    xm = grid.r[-1] * np.cos(thetac)
                    ym = grid.r[-1] * np.sin(thetac)
                    axs[0].plot(xm, ym, '-', linewidth=0.5, color='k')
                    if lxx == lx:
                        axs[0].plot(x0[lx, lxx, rplot, :], y0[lx, lxx, rplot, :], linestyle='solid',
                                    color=color)
                    elif lxx == lx - 1:
                        axs[0].plot(x0[lx, lxx, rplot, :], y0[lx, lxx, rplot, :], linestyle='dashed',
                                    color=color)
                    else:
                        axs[0].plot(x0[lx, lxx, rplot, :], y0[lx, lxx, rplot, :], linestyle='dotted',
                                    color=color)
                    axs[0].plot(xplot, yplot, 'o', color=color)
                elif lx == 2:
                    axs[1].set_title('Selected CP-surfaces for ' + '{}'.format(wave_1[2]), fontsize='small')
                    axs[1].set_xlim(0.0, prm.rmax)
                    axs[1].set_ylim(0.0, prm.rmax)
                    axs[1].set_box_aspect(1.0)
                    axs[1].set_xlabel('$r/R_c$', fontsize='small')
                    # axs[1].set_ylabel('$r/R_c$', fontsize='small')
                    axs[1].tick_params(labelcolor='b', labelsize='8', width=1)
                    axs[1].tick_params(labelcolor='b', labelsize='8', width=1)

                    xc = np.cos(thetac)
                    yc = np.sin(thetac)
                    axs[1].plot(xc, yc, '-', linewidth=0.5, color='k')
                    xm = grid.r[-1] * np.cos(thetac)
                    ym = grid.r[-1] * np.sin(thetac)
                    axs[1].plot(xm, ym, '-', linewidth=0.5, color='k')
                    if lxx == lx:
                        axs[1].plot(x0[lx, lxx, rplot, :], y0[lx, lxx, rplot, :], linestyle='solid',
                                    color=color)
                    elif lxx == lx - 1:
                        axs[1].plot(x0[lx, lxx, rplot, :], y0[lx, lxx, rplot, :], linestyle='dashed',
                                    color=color)
                    else:
                        axs[1].plot(x0[lx, lxx, rplot, :], y0[lx, lxx, rplot, :], linestyle='dotted',
                                    color=color)
                    axs[1].plot(xplot, yplot, 'o', color=color)
                else:
                    pass


# -----------------------------------------------------------
def plot_profiles(lambda0, nfreq, nfreq_tot, dvel, flux, fc):
    # -------------------------------------------------------
    """
    resamples and plots the emergent flux for 3 lines and their sum. Resampling and smoothing is useful to reduce
    the high frequency numerical noise that occurs in some velocity fields
    """
    fig, ax = plt.subplots()
    region = 'all'

    resampled_x_axis, dwave, resampled_flux, blend, average_fc = lib.resample_flux(region, lambda0, nfreq,
                                                                                   nfreq_tot, dvel, flux, fc)
    max_blend = np.max(blend) / average_fc

    # fig.suptitle('Emergent line flux', fontsize='medium')
    ax.set_ylim(-0.1, max_blend + 0.2)
    ax.set_box_aspect(1)
    ax.set_xlim(wav_limits)

    for lx in range(prm.nline):
        ax.plot(resampled_x_axis, resampled_flux[lx] / average_fc, '-', linewidth=1.0,
                color=line_color[lx], label=lab_2[lx])

    ax.plot(resampled_x_axis, blend / average_fc, '-', color='m', linewidth=1.5, label='blend')

    for lx in range(prm.nline):
        ax.plot(lambda0[lx] / const.angstrom, -0.05, '|', color=line_color[lx])

    ax.set_xlabel(r'$\lambda (\AA)$', fontsize='small')
    ax.set_ylabel('$F_\lambda/F_c$', fontsize='small')
    ax.tick_params(labelcolor='b', labelsize='8', width=1, which='both', labelbottom=True)
    ax.legend(prop={'size': 8}, loc='best')


# -----------------------------------------------------------------
def plot_fluo_and_blend(lambda0, nfreq, nfreq_tot, dvel, flux, fc):
    # -------------------------------------------------------------
    """
    plots the FeI line and the overall blend of the 3 lines around 3969A.
    """
    fig, ax = plt.subplots()
    region = 'all'

    resampled_x_axis, dwave, resampled_flux, blend, average_fc = lib.resample_flux(region, lambda0, nfreq,
                                                                                   nfreq_tot, dvel, flux, fc)
    max_blend = np.max(blend) / average_fc

    # fig.suptitle('Emergent line flux', fontsize='medium')
    ax.set_ylim(-0.1, max_blend + 0.2)
    ax.set_box_aspect(1)
    ax.set_xlim(wav_limits)

    ax.plot(resampled_x_axis, resampled_flux[1] / average_fc, '-', linewidth=1.0,
            color=line_color[1], label=lab_2[1])

    ax.plot(resampled_x_axis, blend / average_fc, '-', color='m', linewidth=1.5, label='blend')

    ax.plot(lambda0[1] / const.angstrom, -0.05, '|', color=line_color[1])

    ax.set_xlabel(r'$\lambda (\AA)$', fontsize='small')
    ax.set_ylabel('$F_\lambda/F_c$', fontsize='small')
    ax.tick_params(labelcolor='b', labelsize='8', width=1, which='both', labelbottom=True)
    ax.legend(prop={'size': 8}, loc='upper right')


# ------------------------------------------
def read_and_plot_3_cp_surfaces(in_file_cp):
    # --------------------------------------
    """
    similar to read_and_plot_2_cp_surfaces but for velocity fields leading to self-cp surfaces
    """
    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(24 * cm, 9 * cm))
    # fig.suptitle(r'Common-point surfaces', fontsize='large')

    # --------------------------
    # read and plot cp-surfaces
    # --------------------------
    # (stored in cartesian coordinates)
    x0 = np.zeros((prm.nline, prm.nline, prm.idr, prm.igauss_shell))
    y0 = np.zeros((prm.nline, prm.nline, prm.idr, prm.igauss_shell))
    thetac = [np.deg2rad(t) for t in range(0, 91)]
    with open(in_file_cp, "r") as f:
        while True:
            one_line = f.readline()
            if not one_line:
                break
            v0, v1, v2, v3, v4, v5, v6 = one_line.split()
            rplot = int(v0)
            color = v1
            igauss = int(v2)
            xplot = float(v3)
            yplot = float(v4)
            lx = int(v5)
            lxx = int(v6)

            for j in range(igauss):
                xvalue, yvalue = f.readline().split()
                if float(xvalue) or float(yvalue):
                    x0[lx, lxx, rplot, j] = float(xvalue)
                    y0[lx, lxx, rplot, j] = float(yvalue)
                else:
                    x0[lx, lxx, rplot, j] = None
                    y0[lx, lxx, rplot, j] = None
                # print(x0[lx, lxx, rplot, j], y0[lx, lxx, rplot, j])

            if not np.any(x0[lx, lxx, rplot, :]):
                # print('no cp-surface for this combination')
                pass
            else:
                if lx == 0:
                    axs[0].set_title('Selected CP-surfaces for line ' + lab_1[0], fontsize='small')
                    axs[0].set_xlim(0.0, prm.rmax)
                    axs[0].set_ylim(0.0, prm.rmax)
                    axs[0].set_box_aspect(1.0)
                    axs[0].set_xlabel('$r/R_c$', fontsize='small')
                    axs[0].set_ylabel('$r/R_c$', fontsize='small')
                    axs[0].tick_params(labelcolor='b', labelsize='8', width=1)
                    axs[0].tick_params(labelcolor='b', labelsize='8', width=1)
                    # axs[0].text(7, 9, 'CP-surfaces\nfor $\lambda$3970', color='grey', fontsize=8)
                    xc = np.cos(thetac)
                    yc = np.sin(thetac)
                    axs[0].plot(xc, yc, '-', linewidth=0.5, color='k')
                    xm = grid.r[-1] * np.cos(thetac)
                    ym = grid.r[-1] * np.sin(thetac)
                    axs[0].plot(xm, ym, '-', linewidth=0.5, color='k')
                    if lxx == lx:
                        axs[0].plot(x0[lx, lxx, rplot, :], y0[lx, lxx, rplot, :], linestyle='solid',
                                    color=color)
                    elif lxx == lx + 1:
                        axs[0].plot(x0[lx, lxx, rplot, :], y0[lx, lxx, rplot, :], linestyle='dashed',
                                    color=color)
                    else:
                        axs[0].plot(x0[lx, lxx, rplot, :], y0[lx, lxx, rplot, :], linestyle='dotted',
                                    color=color)
                    axs[0].plot(xplot, yplot, 'o', color=color)
                elif lx == 1:
                    axs[1].set_title('Selected CP-surfaces for line ' + lab_1[1], fontsize='small')
                    axs[1].set_xlim(0.0, prm.rmax)
                    axs[1].set_ylim(0.0, prm.rmax)
                    axs[1].set_box_aspect(1.0)
                    axs[1].set_xlabel('$r/R_c$', fontsize='small')
                    # axs[1].set_ylabel('$r/R_c$', fontsize='small')
                    axs[1].tick_params(labelcolor='b', labelsize='8', width=1)
                    axs[1].tick_params(labelcolor='b', labelsize='8', width=1)
                    # axs[1].text(7, 9, 'CP-surfaces\nfor $\lambda$3970', color='grey', fontsize=8)
                    xc = np.cos(thetac)
                    yc = np.sin(thetac)
                    axs[1].plot(xc, yc, '-', linewidth=0.5, color='k')
                    xm = grid.r[-1] * np.cos(thetac)
                    ym = grid.r[-1] * np.sin(thetac)
                    axs[1].plot(xm, ym, '-', linewidth=0.5, color='k')
                    if lxx == lx:
                        axs[1].plot(x0[lx, lxx, rplot, :], y0[lx, lxx, rplot, :], linestyle='solid',
                                    color=color)
                    elif lxx == lx - 1:
                        axs[1].plot(x0[lx, lxx, rplot, :], y0[lx, lxx, rplot, :], linestyle='dashed',
                                    color=color)
                    else:
                        axs[1].plot(x0[lx, lxx, rplot, :], y0[lx, lxx, rplot, :], linestyle='dotted',
                                    color=color)
                    axs[1].plot(xplot, yplot, 'o', color=color)
                elif lx == 2:
                    axs[2].set_title('Selected CP-surfaces for line ' + lab_1[2], fontsize='small')
                    axs[2].set_xlim(0.0, prm.rmax)
                    axs[2].set_ylim(0.0, prm.rmax)
                    axs[2].set_box_aspect(1.0)
                    axs[2].set_xlabel('$r/R_c$', fontsize='small')
                    # axs[1].set_ylabel('$r/R_c$', fontsize='small')
                    axs[2].tick_params(labelcolor='b', labelsize='8', width=1)
                    axs[2].tick_params(labelcolor='b', labelsize='8', width=1)
                    # axs[1].text(7, 9, 'CP-surfaces\nfor $\lambda$3970', color='grey', fontsize=8)
                    xc = np.cos(thetac)
                    yc = np.sin(thetac)
                    axs[2].plot(xc, yc, '-', linewidth=0.5, color='k')
                    xm = grid.r[-1] * np.cos(thetac)
                    ym = grid.r[-1] * np.sin(thetac)
                    axs[2].plot(xm, ym, '-', linewidth=0.5, color='k')
                    if lxx == lx:
                        axs[2].plot(x0[lx, lxx, rplot, :], y0[lx, lxx, rplot, :], linestyle='solid',
                                    color=color)
                    elif lxx == lx - 1:
                        axs[2].plot(x0[lx, lxx, rplot, :], y0[lx, lxx, rplot, :], linestyle='dashed',
                                    color=color)
                    else:
                        axs[2].plot(x0[lx, lxx, rplot, :], y0[lx, lxx, rplot, :], linestyle='dotted',
                                    color=color)
                    axs[2].plot(xplot, yplot, 'o', color=color)
                else:
                    pass


