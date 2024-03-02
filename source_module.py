"""
module name: source_module
Computes the Sobolev generalized source functions. Some fairly general functions needed in this module
(derq, Q0, vrad...) are stored in the function_library module but could be included here instead.
"""
# ----- import python libraries -----

import numpy as np

# ----- import project modules -----

import function_library as lib
import import_parameters as prm
import set_grids as grid
import in_out_library as io
import plot_library as plres

global eta, betac_nl

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


# -----------------------------------------------------------------------------------------
def escape_gauss_integrals(igauss, wgauss, alim, blim, stheta, ctheta, taur, exp_tau_term):
    # -------------------------------------------------------------------------------------
    """
    computes the escape probability integrals that appear in the local source function,
    using a Gaussian integration scheme. Called by escape
    """
    # loop over all radial grid points
    # initialize auxiliary array
    esc = np.zeros((prm.nline, prm.idr))
    rpz = np.zeros((prm.idr, igauss))
    for i in range(prm.idr):
        rpz[i, :] = grid.r[i]

    dr = prm.frac * rpz
    zr = rpz
    pr = 0.0
    prp = stheta * dr
    zrp = rpz + ctheta * dr
    q0 = 1.0 / lib.derq(zr, zrp, pr, prp, dr, stheta, ctheta)

    for lx in range(prm.nline):
        for i in range(prm.idr):
            esc[lx, i] = 0.0
            s1 = np.zeros(igauss)
            # gauss_int = 0.0
            for j in range(igauss):
                taux = taur[lx, i] * q0[i, j]
                f1 = (1.0 - np.exp(-taux)) / taux
                s1[j] = f1 * wgauss[j] * exp_tau_term[lx, i, j]
            gauss_int = sum(s1)
            esc[lx, i] = 0.5 * gauss_int * (blim[i] - alim[i])

        if prm.source_debug_mode:
            print(f"np.shape(esc) = {np.shape(esc)}")
    # the 1/2 factor below appears in front of the integral over space when assuming spherical symmetry
    return 0.5 * esc


# --------------------------------------
def escape(taur, exp_tau_term, in_core):
    # ----------------------------------
    """
    sets up a Gaussian integration of escape probabilities beta, betac and beta_nl:
    - if in_core = False, return escape probabilities in all directions for all r's;
    - if in_core = True, return probability for a photon to hit the stellar core.
    a and b are the integration limits, xg and wg the Gaussian coordinates and weights
    """

    if in_core:
        igauss = prm.igauss_core
        # cos of angle under which the core is seen from r
        b_core = np.sqrt(1.0 - prm.rc * prm.rc / (grid.r * grid.r))
        core_angle = np.pi - np.arccos(b_core)
        b_core = np.cos(core_angle)

        a = - np.ones(prm.idr)  # lower integration limit
        b = b_core  # upper integration limit

        # reverse_flag = True to have increasing thetas
        cos_theta, wg = lib.ctheta_grid(True, igauss, a, b)

    elif not in_core:
        igauss = prm.igauss_shell
        a = - np.ones(prm.idr)  # grid.a_core - lower integration limit
        b = np.ones(prm.idr)  # upper integration limit
        # reverse_flag = True to have increasing thetas
        cos_theta, wg = lib.ctheta_grid(True, igauss, a, b)

    else:
        print(f"Logical problem in escape; in_core = {in_core}. Abort program")
        exit()
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    theta = np.arccos(cos_theta)  # single-valued over [0, pi]

    esc = escape_gauss_integrals(igauss, wg, a, b, sin_theta, cos_theta, taur, exp_tau_term)

    return esc


# --------------------------------------------------------
def compute_local_source_function(epsilon, bnu, fc, taur):
    # ----------------------------------------------------
    """
    computes the local Sobolev source function
    """
    exp_tau_term = np.ones((prm.nline, prm.idr, prm.igauss_shell))
    source = np.zeros((prm.nline, prm.idr))  # source function vector
    # beta = np.zeros((prm.nline, prm.idr))  # photon escape probability vector
    in_core = False
    beta = escape(taur, exp_tau_term, in_core)
    print(f"beta\n {beta}") if prm.source_debug_mode else None

    # betac = np.zeros(prm.idr)  # continuous stellar core contribution to source
    exp_tau_term = np.ones((prm.nline, prm.idr, prm.igauss_core))
    in_core = True
    betac = escape(taur, exp_tau_term, in_core)
    print(f"betac\n {betac}") if prm.source_debug_mode else None

    if prm.graph_mode:
        plres.plot_beta(beta, betac, 'beta', 'betac')

    for lx in range(prm.nline):
         source[lx, :] = ((1.0 - epsilon[lx, :]) * betac[lx, :] * fc[lx] + epsilon[lx, :] * bnu[lx, :]) \
                        / (epsilon[lx, :] + beta[lx, :] * (1.0 - epsilon[lx, :]))

    if prm.graph_mode:
        plres.plot_l_source(source, fc)

    return source, beta, betac


# -----------------------------------------
def intersections(igauss, r_vector, theta):
    # -------------------------------------
    """
    Computes intersections of rays with angle theta originating
    from (x = r_vector, y = 0) with a shell of radius r_shell.
    x[len(r), igauss, 2], y[len(r),igauss, 2] are the intersection coordinates
    The 3rd index is 0 or 1 for roots corresponding respectively to + or - sign
    in the root quadratic equation
    """
    print('Computing intersections with boundaries') if prm.source_debug_mode else None

    x = np.zeros((len(r_vector), igauss, 2))
    y = np.zeros((len(r_vector), igauss, 2))

    tgtheta = np.tan(theta)
    tg2theta = np.power(tgtheta, 2)

    bterm = (np.zeros((len(r_vector), igauss)).T + r_vector).T * tg2theta
    aterm = (np.zeros((len(r_vector), igauss)).T + 1.0).T + tg2theta
    dterm_1 = (np.zeros((len(r_vector), igauss)).T + r_vector[-1] ** 2).T * aterm
    dterm_2 = (np.zeros((len(r_vector), igauss)).T - r_vector ** 2).T * tg2theta
    dterm = np.sqrt(dterm_1 + dterm_2)

    # Determinant dterm is always > 0 : 2 real roots x, y for each ray
    # first root: x[:,:,0]
    xtemp = (bterm + dterm) / aterm
    x[:, :, 0] = xtemp

    # second root: x[:,:,1]
    xtemp = (bterm - dterm) / aterm
    x[:, :, 1] = xtemp

    # corresponding y's
    y_1 = np.ones((len(r_vector), igauss)) * tgtheta
    y_2 = (np.zeros((len(r_vector), igauss)).T + r_vector).T - x[:, :, 0]
    y[:, :, 0] = y_1 * y_2
    y_3 = (np.zeros((len(r_vector), igauss)).T + r_vector).T - x[:, :, 1]
    y[:, :, 1] = y_1 * y_3
    print('Exiting intersections') if prm.source_debug_mode else None

    return x, y


# ----------------------------------------------------
def distance_to_border(igauss, stheta, ctheta, theta):
    # ------------------------------------------------
    """
    computes the distances from r to the core or the shell in the directions defined by theta
    CAUTION:  we assume spherical symmetry here, and restrict ray tracing to one root direction
              taking farthest root (1) for theta > pi/2, closest one (0) for theta < pi/2
    """

    distance = np.zeros((prm.idr, igauss))
    print('Computing distances to boundaries') if prm.source_debug_mode else None
    x, y = intersections(igauss, grid.r, theta)

    x1 = np.where(ctheta <= 0, x[:, :, 1], x[:, :, 0])
    y1 = np.where(ctheta <= 0, y[:, :, 1], y[:, :, 0])

    # eliminate rounding errors near x1, y1 = 0.0
    x1 = np.where(np.abs(x1) < 1.0e-5, 0.0, x1)
    y1 = np.where(np.abs(y1) < 1.0e-5, 0.0, y1)

    b_core = np.sqrt(1.0 - prm.rc * prm.rc / (grid.r * grid.r))
    core_angle = np.pi - np.arccos(b_core)

    for i in range(prm.idr):
        for j in range(igauss):
            if theta[i, j] > core_angle[i]:
                dist_squared = prm.rc ** 2 - 2 * grid.r[i] ** 2 * stheta[i, j] ** 2 + grid.r[i] ** 2 - \
                               2 * grid.r[i] * np.sqrt(
                    (prm.rc ** 2 - grid.r[i] ** 2 * stheta[i, j] ** 2) * ctheta[i, j] ** 2)
            else:
                dist_squared = (x1[i, j] - grid.r[i]) ** 2 + y1[i, j] ** 2
            if dist_squared <= prm.frac:
                distance[i, j] = 0.0
            else:
                distance[i, j] = np.sqrt(dist_squared)
            if prm.source_debug_mode:
                print(f"theta[{i}, {j}] = {np.rad2deg(theta[i, j])}, distance[{i}, {j}] = {distance[i, j]}")
    print('Exiting distances') if prm.source_debug_mode else None
    return distance


# ------------
def rg2rl(igauss, stheta, ctheta, rg):
    # --------
    # rg's are defined along the ray lx. We must now go back to the (r, theta) frame by
    # computing the corresponding xl,yl along ray lx and from there rl and v(rl)
    # in (r, theta) frame. We also compute rl + drl for computing the vrad derivative
    print('Entering rg2rl') if prm.source_debug_mode else None
    rl = np.zeros((prm.idr, igauss), object)
    xl = np.zeros((prm.idr, igauss), object)
    yl = np.zeros((prm.idr, igauss), object)
    xg = np.zeros((prm.idr, igauss), object)
    yg = np.zeros((prm.idr, igauss), object)

    for i in range(prm.idr):
        xg[i, :] = rg[i, :] * ctheta[i, :]
        yg[i, :] = rg[i, :] * stheta[i, :]
        if prm.source_debug_mode:
            print(f"xg[{i}, :]\n{xg[i, :]}\nyg[{i}, :]\n{yg[i, :]}\n")

    for i in range(prm.idr):
        xl[i, :] = xg[i, :] + grid.r[i]
        yl[i, :] = yg[i, :]

    if prm.graph_mode:
        for rplot, rsymb in grid.rplots.items():
            plres.plot_rl(igauss, rplot, rsymb, xl, yl)

    rl2 = xl ** 2 + yl ** 2

    for i in range(prm.idr):
        for j in range(igauss):
            rl[i, j] = np.sqrt(rl2[i, j])

        if prm.source_debug_mode:
            print(f"----- i = {i} ----- \nxl[{i}, :]\n{xl[i, :]}\nyl[{i}, :]\n{yl[i, :]}\nrl[{i}, :]\n{rl[i, :]}\n")
    print('Exiting rg2rl') if prm.source_debug_mode else None

    return xl, yl, rl


# ----------------------------------------------------
def rad_vel_over_space(igauss, stheta, ctheta, theta):
    # ------------------------------------------------
    # Find and store for future use the geometry of CP surfaces
    # as seen from all radial grid points and all rays.

    # Define angular grid for the Gaussian integration over all space
    # see eq. 47 in Rybicky and Hummer 1978
    print('Computing radial velocities over space') if prm.source_debug_mode else None
    distance = distance_to_border(igauss, stheta, ctheta, theta)
    dist_lim = 0.1
    distance = np.where(distance > dist_lim, distance, 0.0)

    # transform distances to vectors (from 0 to distance[], in int_dist+2 stprm.eps)
    # rg is the matrix of radii along the rays (rd, theta)

    int_dist = np.zeros((prm.idr, igauss), dtype=int)
    rg = np.ones((prm.idr, igauss), object)
    for i in range(prm.idr):
        int_dist[i, :] = np.round(distance[i, :]).astype(int)
        for j in range(igauss):
            rg[i, j] = np.linspace(0.0, distance[i, j], 2 + prm.idz * int_dist[i, j])

    if prm.source_debug_mode:
        print(f"Integer distance from P to r_shell or prm.rc along ray \n {int_dist[:, :]}")
        print(f"rg \n {rg[:, :]}")

    # rg's are defined along the rays (rd, theta). We must now go back to the (r, theta) frame by
    # computing the corresponding xl,yl along rays and from there rl and v(rl)
    # in (r, theta) frame. We also compute rl + drl for computing the vrad derivative

    xl, yl, rl = rg2rl(igauss, stheta, ctheta, rg)

    vrad_on_ray = np.zeros((prm.idr, igauss), object)
    vrad0 = np.zeros((prm.idr, igauss))
    sth = yl / rl
    cth = xl / rl

    print('Entering vrad_on_ray loop: compute vrad variation on all rays')
    # import time
    """
    start_time = time.time()
    for i in range(prm.idr):
        for j in range(igauss):
            prms = [(rl[i, j][k], sth[i, j][k], cth[i, j][k], stheta[i, j], ctheta[i, j])
                    for k in range(len(rl[i, j]))]
            vrad_on_ray[i, j] = list(map(lib.vrads_0, prms))
    exec_time = time.time() - start_time
    print(f"exec_time = {exec_time}")
    """
    # start_time = time.time()
    for i in range(prm.idr):
        for j in range(igauss):
            vrad_on_ray[i, j] = lib.vrads(rl[i, j], sth[i, j], cth[i, j], stheta[i, j], ctheta[i, j])
            if prm.source_debug_mode:
                vrad0[i, j] = grid.v_law(grid.r[i]) * ctheta[i, j]
                if np.abs(vrad0[i, j] - prm.eps) < vrad_on_ray[i, j][0] < np.abs(vrad0[i, j] + prm.eps):
                    print(f"vrad_on_ray[{i}, {j}][0] = {vrad_on_ray[i, j][0]} != vrad0[{i}, {j}] = {vrad0[i, j]}")
                    print('abort program')
                    exit()
    # exec_time = time.time() - start_time
    # print(f"exec_time = {exec_time}")

    print('Exiting vrad_on_ray loop') if prm.source_debug_mode else None
    if prm.graph_mode:
        for rplot, rsymb in grid.rplots.items():
            plres.plot_vrad(igauss, rplot, xl, yl, vrad_on_ray)

    return xl, yl, rl, vrad_on_ray


# ----------------------------------------------------------------------------
def locate_cp_surfaces(igauss, lx, lxx, delta_v_mat, vrad_on_ray, xl, yl, rl):
    # ------------------------------------------------------------------------
    """
    from radial velocities found on the rays over all space,
    find the approximate location of the CP surfaces,
    and the value of dvrad/dl at that location
    """
    print(f"Computing CP surfaces for lx = {lx}, lxx = {lxx}...")
    r0 = np.zeros((prm.nline, prm.nline, prm.idr, igauss))
    x0 = np.zeros((prm.nline, prm.nline, prm.idr, igauss))
    y0 = np.zeros((prm.nline, prm.nline, prm.idr, igauss))
    one_over_dvrad0 = np.ones((prm.nline, prm.nline, prm.idr, igauss)) * prm.frac

    vrad_on_ray_dif = np.zeros((prm.idr, igauss), object)
    vrad_sign = np.zeros((prm.idr, igauss), object)
    cp_surface_loc = np.zeros((prm.idr, igauss), dtype=np.integer)

    # CP surface intersects ray at zero relative radial velocity
    # find its approximate location by looking for a change of sign in vrad
    print(f"enter CP-locate with lx = {lx}, lxx = {lxx}, delta_v = {delta_v_mat[lx, lxx]}") \
        if prm.source_debug_mode else None

    for i in range(prm.idr):
        for j in range(igauss):
            if prm.interact_mode:
                # ----------------------------------------------------------------
                vrad_on_ray_dif[i, j] = vrad_on_ray[i, j] - vrad_on_ray[i, j][0] \
                                        - delta_v_mat[lx, lxx]
                # ----------------------------------------------------------------
            else:
                vrad_on_ray_dif[i, j] = vrad_on_ray[i, j] - vrad_on_ray[i, j][0]
            # --------------------------------------------------------------------
            vrad_sign[i, j] = lib.find_sign_change(vrad_on_ray_dif[i, j])
            cp_surface_loc[i, j] = np.argmax(vrad_sign[i, j], axis=0)
            if cp_surface_loc[i, j]:
                rl_index_m = cp_surface_loc[i, j] - 1
                rl_index_p = cp_surface_loc[i, j]
                rl_loc_m = rl[i, j][rl_index_m]
                rl_loc_p = rl[i, j][rl_index_p]
                xl_loc_m = xl[i, j][rl_index_m]
                xl_loc_p = xl[i, j][rl_index_p]
                yl_loc_m = yl[i, j][rl_index_m]
                yl_loc_p = yl[i, j][rl_index_p]
                vr_loc_m = vrad_on_ray_dif[i, j][rl_index_m]
                vr_loc_p = vrad_on_ray_dif[i, j][rl_index_p]
                if prm.source_debug_mode:
                    print(f"cp_surface_loc[{i},{j}] is True")
                    print(f"rl_index_m, rl_index_p = {rl_index_m}, {rl_index_p}")
                    print(f"rl_loc_m, rl_loc_p = {rl_loc_m}, {rl_loc_p}")
                    print(f"vr_loc_m, vr_loc_p = {vr_loc_m}, {vr_loc_p}")

                # perform linear interpolation in rl to find CP surface intersection with ray
                r0[lx, lxx, i, j] = (rl_loc_m * vr_loc_p - rl_loc_p * vr_loc_m) / (vr_loc_p - vr_loc_m)
                # derive approximate locations for x0 and y0, used for graphs only
                x0[lx, lxx, i, j] = (xl_loc_m * vr_loc_p - xl_loc_p * vr_loc_m) / (vr_loc_p - vr_loc_m)
                y0[lx, lxx, i, j] = (yl_loc_m * vr_loc_p - yl_loc_p * vr_loc_m) / (vr_loc_p - vr_loc_m)
                # and compute abs(dvrad/dl) at the same location (recall that vrad = 0 at r0)
                dr = rl_loc_p - r0[lx, lxx, i, j]
                if not dr:
                    dr = prm.frac
                dvrad = np.abs(vr_loc_p / dr)
                one_over_dvrad0[lx, lxx, i, j] = 1.0 / dvrad

                if prm.source_debug_mode:
                    print(f"ro[{lx}, {lxx}, {i}, {j}] = {r0[lx, lxx, i, j]}")
                    print(f"one_over_dvrad0[{lx}, {lxx}, {i}, {j}] = {one_over_dvrad0[lx, lxx, i, j]}")

    if prm.source_debug_mode:
        print(f"vrad_on_ray_dif \n{vrad_on_ray_dif}")
        print(f"vrad_sign \n{vrad_sign}")
        print(f"vrad_on_ray \n {vrad_on_ray}")
        print(f"cp_surface_loc \n {cp_surface_loc}")
        print(f"from locate CP -- r0[{lx}, {lxx}, :, :] \n {r0[lx, lxx, :, :]}")
        print(f"one_over_dvrad0[{lx}, {lxx}, :, :] \n {one_over_dvrad0[lx, lxx, :, :]}")

    if prm.graph_mode:
        plres.plot_r0(igauss, lx, lxx, x0, y0)

    return r0[lx, lxx, :, :], x0[lx, lxx, :, :], y0[lx, lxx, :, :], one_over_dvrad0[lx, lxx, :, :]


# -----------------------------------------------------------------------------
def order_cp_surface_intersections(igauss, r0, x0, y0, one_over_dvrad0):
    # -------------------------------------------------------------------------
    """
    Organizes cp-surface intersections for all lines of sight and all radial points. Relevant data
    are collected in t0_array[i, j, lx] in the form of tuples (line number, r0, q0_nl = one_over_dvrad0 at r0,
    distance from r to r0). The distance is used for sorting the roots and discarded thereafter.
    r0's are stored in order of increasing distance from the (radial) point of computation
    """
    t0_array = np.zeros((prm.idr, igauss, prm.nline), object)
    r0_dist = np.zeros((prm.nline, prm.nline, prm.idr, igauss))
    for i in range(prm.idr):
        for j in range(igauss):
            for lx in range(prm.nline):
                t0_list = []
                for lxx in range(prm.nline):
                    r_cp = r0[lx, lxx, i, j]
                    if r_cp > prm.frac:
                        # CAUTION: since x0 and y0 are only approximate,
                        # the sorting procedure may need to be checked in more detail
                        r0_dist[lx, lxx, i, j] = np.sqrt((x0[lx, lxx, i, j] - grid.r[i]) ** 2 + y0[lx, lxx, i, j] ** 2)
                        r0_dist_s = r0_dist[lx, lxx, i, j]
                        t0_list.append((lxx, r0[lx, lxx, i, j], one_over_dvrad0[lx, lxx, i, j], r0_dist_s))
                print(f"i = {i}, j = {j}, lx = {lx}, t0_list = {t0_list}") if prm.source_debug_mode else None

                if len(t0_list) > 1:
                    t0_list_sorted = [(k, v, w, t) for k, v, w, t in sorted(t0_list, key=lambda item: item[3],
                                                                            reverse=False)]
                    print(f"sorted t0_list = {t0_list_sorted}") if prm.source_debug_mode else None
                    t0_array[i, j, lx] = [(k, v, w) for (k, v, w, t) in t0_list_sorted]
                elif len(t0_list) == 1:
                    t0_array[i, j, lx] = [(k, v, w) for (k, v, w, t) in t0_list]
                print(f"t0_array[{i}, {j}, {lx}] = {t0_array[i, j, lx]}") if prm.source_debug_mode else None

    if prm.source_debug_mode:
        for i in range(prm.idr):
            for lx in range(prm.nline):
                print(f"t0_array[{i}, :, {lx}]\n{t0_array[i, :, lx]}")

    return t0_array


# ------------------------------------------------------------------------------------
def find_interacting_surfaces(out_file_cp, delta_v_mat, self_cp, inter_cp, igauss_eta,
                              stheta_eta, ctheta_eta, theta_eta):
    # --------------------------------------------------------------------------------
    """
    find, order and store the interacting surfaces
    """
    t0_array = np.zeros((prm.idr, igauss_eta, prm.nline), object)

    # compute distribution of radial velocities over space
    xl, yl, rl, vrad_on_ray = rad_vel_over_space(igauss_eta, stheta_eta, ctheta_eta, theta_eta)

    if self_cp.any() or inter_cp.any():
        print(f"Entering interaction loop") if prm.source_debug_mode else None
        # vrad_on_ray depends only on envelope velocity field.
        # We now introduce the line displacements to find location of relevant CP surfaces
        # assign arrays for CP determination
        r0 = np.zeros((prm.nline, prm.nline, prm.idr, igauss_eta))
        x0 = np.zeros((prm.nline, prm.nline, prm.idr, igauss_eta))
        y0 = np.zeros((prm.nline, prm.nline, prm.idr, igauss_eta))
        q0 = np.zeros((prm.nline, prm.nline, prm.idr, igauss_eta))
        for lx in range(prm.nline):
            nint = 0
            for lxx in range(prm.nline):
                if self_cp[lx, lxx]:  # self CP in non-monotonic flow
                    print(f"self CP -- lx = {lx}, lxx = {lxx}") if prm.source_debug_mode else None
                    r0[lx, lxx, :, :], x0[lx, lxx, :, :], y0[lx, lxx, :, :], q0[lx, lxx, :, :] \
                        = locate_cp_surfaces(igauss_eta, lx, lxx, delta_v_mat, vrad_on_ray, xl, yl, rl)
                    print(f"approx. CP roots r0[{lx}, {lxx}]\n{r0[lx, lxx, :, :]}") \
                        if prm.source_debug_mode else None
                    print(f"dvrad/dl at r0\n{q0[lx, lxx, :, :]}\n") if prm.source_debug_mode else None
                elif inter_cp[lx, lxx] and prm.interact_mode:  # CP for interacting lines
                    nint += 1
                    print(f"interaction CP #{nint}-- lx = {lx}, lxx = {lxx}") \
                        if prm.source_debug_mode else None
                    r0[lx, lxx, :, :], x0[lx, lxx, :, :], y0[lx, lxx, :, :], q0[lx, lxx, :, :] \
                        = locate_cp_surfaces(igauss_eta, lx, lxx, delta_v_mat, vrad_on_ray, xl, yl, rl)
                    print(f"approx. CP roots r0[{lx}, {lxx}]\n{r0[lx, lxx, :, :]}") \
                        if prm.source_debug_mode else None
                else:
                    pass
        if prm.nline <= 3:
            # save selected cp surfaces for plot purposes
            io.save_cp_surfaces(out_file_cp, grid.rplots, igauss_eta, x0, y0)
        # t0_array[i, j, lx] contains the ordered cp-intersection coordinates along all rays
        # together with values of q0 = 1/dvrad0 at the same points
        t0_array = order_cp_surface_intersections(igauss_eta, r0, x0, y0, q0)

    return t0_array


# --------------------------------------------------------------------------------
def eta_interact_integral(igauss, wgauss, t0_array, local_term, source, taur, fc):
    # ----------------------------------------------------------------------------
    """
    computes the eta integral (including source term) (see eq. 47 of Rybicky and Hummer 1978),
    using the Gaussian approx., for all rd's. q0 is 1/abs(Q) in optical depth expression.
    prm.eps is introduced as minimum r0 value to avoid division by zero in function Q.
    derived Qs for prm.eps are much smaller than prm.eps and can thus be filtered out if needed
    caution: prm.eps must be very small (1e-10 or less) to avoid filtering bona fide Q values
    """

    r0 = np.zeros((prm.nline, prm.nline, prm.idr, igauss))
    source0 = np.zeros((prm.nline, prm.nline, prm.idr, igauss))
    taur0 = np.zeros((prm.nline, prm.nline, prm.idr, igauss))
    exp_taur0 = np.zeros((prm.nline, prm.nline, prm.idr, igauss))
    s_tau_term = np.zeros((prm.nline, prm.nline, prm.idr, igauss))
    # q0_nl = 1 / vrad0 is needed for the non-local optical depth tau_prime at r0
    q0_nl = np.zeros((prm.nline, prm.nline, prm.idr, igauss))

    print(f"\nEntering eta_interact integration\n") if prm.source_debug_mode else None

    # now we evaluate the non-local term(s) to eta. This depends on the number of CP surfaces crossed by the rays
    # we proceed from the considered r-point to the envelope border (or stellar core) and collect contributions to eta
    exp_tau_term = np.ones((prm.nline, prm.idr, igauss))
    non_local_term = np.zeros((prm.nline, prm.idr, igauss))
    for i in range(prm.idr):
        for j in range(igauss):
            for lx in range(prm.nline):
                tau_term = np.zeros(prm.nline)
                if t0_array[i, j, lx]:
                    # -----------------------------------------------------------------
                    # the k-loop goes through the lines interacting with lx, if any.
                    # for each interaction, we compute exp_taur0 = exp(-tau[k]) and
                    # s_tau_term = S[k](1-exp(-tau[k])). To handle k > 0 interactions,
                    # we store tau[k] in tau_term and multiply the current s_tau_term
                    # by exp(-tau[k-1]). The final exp_tau_term = exp(sum(-tau[all k]))
                    # is used by the escape library function in the non local mode
                    # so it is defined at module level.
                    # ------------------------------------------------------------------
                    for k in range(len(t0_array[i, j, lx])):
                        lxx = t0_array[i, j, lx][k][0]
                        print(f"--- k = {k}, lxx = {t0_array[i, j, lx][k][0]}") if prm.source_debug_mode else None
                        r0[lx, lxx, i, j] = t0_array[i, j, lx][k][1]
                        q0_nl[lx, lxx, i, j] = t0_array[i, j, lx][k][2]
                        source0[lx, lxx, i, j] = np.interp(r0[lx, lxx, i, j], grid.r[:], source[lx, :])
                        taur0[lx, lxx, i, j] = np.interp(r0[lx, lxx, i, j], grid.r[:], taur[lx, :]) \
                            * q0_nl[lx, lxx, i, j]

                        if prm.source_debug_mode:
                            print(f"t0_array[{i}, {j}, {lx}] = {t0_array[i, j, lx]}")
                            print(f"r0[{lx}, {lxx}, {i}, {j}] = {r0[lx, lxx, i, j]}"
                                  f"  source0[{lx}, {lxx}, {i}, {j}] = {source0[lx, lxx, i, j] / fc[lx]}"
                                  f"  taur0[{lx}, {lxx}, {i}, {j}] = {taur0[lx, lxx, i, j]}"
                                  f"  q0_nl[{lx}, {lxx}, {i}, {j}] = {q0_nl[lx, lxx, i, j]}")

                        exp_taur0[lx, lxx, i, j] = np.exp(-taur0[lx, lxx, i, j])
                        tau_term[k] = taur0[lx, lxx, i, j]
                        print(f"tau_term[{k}] = {tau_term[k]}") if prm.source_debug_mode else None
                        s_tau_term[lx, lxx, i, j] = source0[lx, lxx, i, j] * (1.0 - exp_taur0[lx, lxx, i, j])
                        if prm.source_debug_mode:
                            print(f"exp_taur0[{lx}, {lxx}, {i}, {j}] = {exp_taur0[lx, lxx, i, j]},"
                                  f" s_tau_term[{lx}, {lxx}, {i}, {j}] = {s_tau_term[lx, lxx, i, j]}")
                        if k:
                            s_tau_term[lx, lxx, i, j] *= np.exp(-tau_term[k - 1])
                            if prm.source_debug_mode:
                                print(f"interaction with lxx = {t0_array[i, j, lx][k - 1][0]}"
                                      f"  tau_term[{k - 1}] = {tau_term[k - 1]}"
                                      f"  tau_term[{k}] = {tau_term[k]}"
                                      f"  s_tau_term[{lx}, {lxx}, {i}, {j}] = {s_tau_term[lx, lxx, i, j]}")

                    exp_tau_term[lx, i, j] = np.exp(-tau_term.sum())
                    print(f"exp_tau_term[{lx}, {i}, {j}] = {exp_tau_term[lx, i, j]}") \
                        if prm.source_debug_mode else None
                    non_local_term[lx, i, j] = s_tau_term[lx, :, i, j].sum()

    if prm.source_debug_mode:
        print(f"non_local_term \n {non_local_term}")
        print(f"shape(non_local_term) = {np.shape(non_local_term)}")

    integrand = local_term * non_local_term

    if prm.source_debug_mode:
        print(f"integrand \n {integrand}")
        print(f"shape(integrand) = {np.shape(integrand)}")

    # finally compute the eta integral at each rd by summing over all theta rays
    # The normalization factor in front of the Gaussian integration is (b-a)/2 = 1
    # the Gaussian integration is done in the following loop

    eta_int = np.zeros((prm.nline, prm.idr))
    for lx in range(prm.nline):
        for i in range(prm.idr):
            eta_int[lx, i] = (integrand[lx, i, :] * wgauss[:]).sum()

    if prm.source_debug_mode:
        print(f"eta integral \n {0.5 * eta}")
        print(f"shape(eta) = {np.shape(eta)}")

    # the 1/2 factor below appears in front of the integral over space when assuming spherical symmetry
    # see also escape in library_2020
    return 0.5 * eta_int, exp_tau_term


# -------------------------------------------------------
def compute_eta_local_term(igauss, stheta, ctheta, taur):
    # ---------------------------------------------------
    # first compute the local optical depth term tau_local at r
    # since the atomic level populations do not change in our LTE approximation,
    # this needs be evaluated once and for all.
    # q0_l = np.zeros((prm.idr, grid.igauss))
    tau_local = np.zeros((prm.nline, prm.idr, igauss))

    q0_l = 1.0 / lib.Q_l(igauss, stheta, ctheta)
    # q0_l = np.where(q0_l < prm.eps, 0.0, q0_l)
    if prm.source_debug_mode:
        print(f"q0_l \n{q0_l}")
    print(f"shape(q0_l) = {np.shape(q0_l)}") if prm.source_debug_mode else None
    for lx in range(prm.nline):
        for i in range(prm.idr):
            for j in range(igauss):
                tau_local[lx, i, j] = taur[lx, i] * q0_l[i, j]
        if prm.source_debug_mode:
            print(f"tau_local[{lx}]\n{tau_local[lx, :, :]}")
    print(f"shape(tau_local) = {np.shape(tau_local)}") if prm.source_debug_mode else None
    # compute the local term in the eta integrand
    eta_local_term = np.where(tau_local, (1.0 - np.exp(-tau_local)) / tau_local, 0.0)

    if prm.source_debug_mode:
        print(f"local_term \n {eta_local_term}")
        print(f"shape(local_term) = {np.shape(eta_local_term)}")

    return eta_local_term


def compute_non_local_source_function(out_file_cp, store_cp_flag, use_stored_cp_flag, delta_v_mat, self_cp, inter_cp,
                                      epsilon, bnu, fc, beta, betac, source_l, taur):
    """
    computes the generalized Sobolev source function (cf. Rybicki & Hummer 1978)
    taking into account line self-interactions in decelerating flows and/or
    non-local interactions with neighboring lines in accelerating and decelerating
    velocity fields
    """
    global eta, betac_nl
    # ------------------------------------------------
    # compute the non-local contribution to the source
    # ------------------------------------------------

    source = source_l

    # set up the Gaussian integration grid for eta
    igauss_eta = prm.igauss_shell
    a = - np.ones(prm.idr)
    b = np.ones(prm.idr)

    ctheta_eta, wgauss_eta = lib.ctheta_grid(True, igauss_eta, a, b)
    stheta_eta = np.sqrt(1.0 - ctheta_eta * ctheta_eta)
    theta_eta = np.arccos(ctheta_eta)
    if prm.debug_mode:
        print(f"theta grid for eta integral \n{np.rad2deg(theta_eta)}")

    # t0_array = np.zeros((prm.idr, igauss_eta, prm.nline), object)

    if use_stored_cp_flag:
        # read stored cp geometry
        with open(prm.file_location_base + 'cp_surfaces.npy', 'rb') as infile:
            t0_array = np.load(infile, allow_pickle=True)
            print(f"full CP-set was retrieved")
    else:
        # compute cp geometry
        t0_array = find_interacting_surfaces(out_file_cp, delta_v_mat, self_cp, inter_cp, igauss_eta,
                                             stheta_eta, ctheta_eta, theta_eta)
        if store_cp_flag:
            # store cp-geometry for future use
            with open(prm.file_location_base + 'cp_surfaces.npy', 'wb') as outfile:
                np.save(outfile, t0_array)
                print(f"Full CP-set was stored")
        else:
            pass

    # -----------------------------------------------
    # iterate on non-local contribution to the source
    # -----------------------------------------------

    dif_source = np.ones((prm.nline, prm.idr))  # source difference between two iterations

    eta_local_term = compute_eta_local_term(igauss_eta, stheta_eta, ctheta_eta, taur)
    iteration = 0

    while np.any(np.abs(dif_source / source) > prm.conv):
        iteration += 1
        print(f"Source iteration {iteration}")

        eta, exp_tau_term = eta_interact_integral(igauss_eta, wgauss_eta, t0_array, eta_local_term, source, taur, fc)

        # betac_nl is the core striking probability in the presence of CPs
        in_core = True
        betac_nl = escape(taur, exp_tau_term, in_core)
        print(f"betac_nl\n {betac_nl}") if prm.source_debug_mode else None

        # replace betac by betac_nl in interacting lines
        for lx in range(prm.nline):
            betac[lx] = betac_nl[lx]

        source_nl = np.zeros((prm.nline, prm.idr))
        for lx in range(prm.nline):
            source_nl[lx, :] = ((1.0 - epsilon[lx, :]) * (eta[lx, :] + betac[lx, :] * fc[lx]) +
                                epsilon[lx, :] * bnu[lx, :]) / (beta[lx, :] * (1.0 - epsilon[lx, :])
                                                                + epsilon[lx, :])
        dif_source = source - source_nl
        print(f"iteration {iteration} - source \n {source_nl} \n dif_source (%) \n {dif_source / source}") \
            if prm.source_debug_mode else None

        # new source estimate
        source = source_nl

        if iteration > prm.iconv_max:
            print(f"exit: source iteration doesn't converge.")
            for lx in range(prm.nline):
                print(source[lx, :] / fc[lx])
            exit()

    if prm.graph_mode:
        plres.plot_eta(eta)
        plres.plot_beta(beta, betac_nl, 'beta', 'betac_nl')

    return source, iteration
