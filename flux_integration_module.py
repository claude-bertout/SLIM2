"""
module name: flux_integration_module
library of functions for performing the exact profile integration
The core loop dint() is not that diffferent from the 1984 version... ;-)
"""

import numpy as np
import collections

import function_library as lib
import import_parameters as prm
import set_grids as grid


# --------------------------
def z_grid(region, idp, pj):
    # ----------------------
    """
    sets up the z-grid over the whole set pj of impact parameters
    """
    zpj = np.zeros(idp, object)

    zmax = np.sqrt(grid.r[-1] ** 2 - pj ** 2)
    print(f"zmax \n {zmax}") if prm.flux_debug_mode else None

    if not prm.log_rgrid:
        # linear z-grid
        zmin = np.where(pj >= grid.r[0], -zmax, np.sqrt(np.abs(grid.r[0] ** 2 - pj ** 2)))
        print(f"zmin \n {zmin}") if prm.flux_debug_mode else None
        deltaz = zmax - zmin
        ideltaz = np.round(deltaz).astype(int)
        print(f"deltaz \n {deltaz}") if prm.flux_debug_mode else None
        print(f"ideltaz \n {ideltaz}") if prm.flux_debug_mode else None
        for i in range(idp):
            zpj[i] = np.linspace(zmin[i], zmax[i], prm.idz * ideltaz[i] + 2)
    else:
        # logarithmic z-grid
        if region == 'core':
            zmin = np.sqrt(np.abs(grid.r[0] ** 2 - pj ** 2))
            print(f"zmin \n {zmin}") if prm.flux_debug_mode else None
            dzmin = np.log(zmin)
            dzmax = np.log(zmax)
            deltaz = dzmax - dzmin
            ideltaz = np.round(deltaz).astype(int)
            for i in range(idp):
                zpj[i] = np.exp(np.linspace(dzmin[i], dzmax[i], prm.idz * ideltaz[i] + 2))
        elif region == 'envelope':
            zmin = np.ones(idp) * prm.frac
            print(f"zmin \n {zmin}") if prm.flux_debug_mode else None
            dzmin = np.log(zmin)
            dzmax = np.log(zmax)
            print(f"dzmin \n {dzmin}") if prm.flux_debug_mode else None
            print(f"dzmax \n {dzmax}") if prm.flux_debug_mode else None
            deltaz = dzmax - dzmin
            ideltaz = np.round(deltaz).astype(int)
            print(f"deltaz \n {deltaz}") if prm.flux_debug_mode else None
            print(f"ideltaz \n {ideltaz}") if prm.flux_debug_mode else None
            for i in range(idp):
                zone1 = np.exp(np.linspace(dzmin[i], dzmax[i], prm.idz * ideltaz[i] + 2))
                # print(f"zone1 \n {zone1}")  #if prm.flux_debug_mode else None
                zone0 = zone1[::-1]
                zone2 = - zone0
                zpj[i] = np.concatenate((zone2, zone1))
        else:
            print('region error in z_grid function. Abort program.')
            exit()
    print(f"zpj \n {zpj}") if prm.flux_debug_mode else None

    # corresponding radii for computing radially symmetric quantities
    rzpj = np.zeros(idp, object)
    for i in range(idp):
        rzpj[i] = np.sqrt(pj[i] ** 2 + zpj[i] ** 2)
    print(f"rzpj \n {rzpj}") if prm.flux_debug_mode else None

    return zpj, rzpj


# -----------------------------------------------------------
def freq_grid(idp, p, z, vr, freq_dict, nwing, f_min, f_max):
    # -------------------------------------------------------
    """
    obtains the z's corresponding to the frequency displacements
    found on the impact parameters, using a linear interpolation between neighboring points.
    Tests with interp1d, which allows for non-monotonical abscissa points, gave wrong results
    hence this slightly convoluted method, which is robust even for complex velocity fields.
    Increase the number of z-points to improve accuracy as needed.
    freq_grid is valid for monotonic and non-monotonic velocity fields.
    All quantities returned here are envelope properties, and thus line-independent
    """
    freq_dict_on_z = {}
    kz_array = np.zeros(idp, object)
    fz_array = np.zeros(idp, object)
    z_interp = np.zeros(idp, object)
    r_interp = np.zeros(idp, object)
    kz_ext = np.zeros(idp, object)

    for i in range(idp):
        print(f"\nFrom freq_grid ----- impact parameter i = {i}") if prm.flux_debug_mode else None
        # find all frequency points on the ray
        freq_dict_on_z = {k: v for k, v in freq_dict.items() if (f_min[i] <= v <= f_max[i])}
        print(f"freq_dict_on_z \n {freq_dict_on_z}") if prm.flux_debug_mode else None

        fz = []
        z_conv = []
        kz = []
        # associate in a tuple the z's to their respective radial velocities on the ray
        pair = [(z[i][j], vr[i][j]) for j in range(len(z[i]))]
        if prm.flux_debug_mode:
            print(f"pair (z, vr) \n {pair}")

        for idx in range(1, len(pair)):
            # order the pair in increasing v
            pair1 = pair[idx - 1]
            pair2 = pair[idx]
            pairmax = pair2 if pair2[1] > pair1[1] else pair1
            pairmin = pair1 if pair1[1] < pair2[1] else pair2

            # interpolate to find z's corresponding to the kz,fz's
            for f_key, f_value in freq_dict_on_z.items():
                if pairmin[1] <= f_value < pairmax[1]:
                    kz.append(f_key)
                    fz.append(f_value)
                    z_conv.append(np.interp(f_value, [pairmin[1], pairmax[1]], [pairmin[0], pairmax[0]]))

        print(f"len(z_conv) = {len(z_conv)}, len(kz) = {len(kz)}, len(fz) = {len(fz)}") \
            if prm.flux_debug_mode else None

        # order with increasing z's
        z_conv, kz, fz = zip(*sorted(zip(z_conv, kz, fz)))

        z_interp[i] = np.array(z_conv)
        r_interp[i] = np.sqrt(z_interp[i] * z_interp[i] + p[i] * p[i])
        kz_array[i] = np.array(kz)
        fz_array[i] = np.array(fz)

        # Add line wings to kz_ext (but not to kz_array,
        # which indexes velocities actually found in envelope)
        kz_ext[i] = sorted(list(collections.OrderedDict.fromkeys(kz_array[i])))
        kz_ext[i] = [j for j in range(kz_ext[i][0] - nwing, kz_ext[i][0])] + kz_ext[i] \
            + [j for j in range(kz_ext[i][-1] + 1, kz_ext[i][-1] + nwing + 1)]
        print(f"extended kz \n {kz_ext[i]}") if prm.flux_debug_mode else None
        print(f"r_interp[{i}] \n{r_interp[i]}") if prm.flux_debug_mode else None

        # the flux integration is done from zmax to zmin, so reorder z and kz, kz_vel.
        # Caution: this must be done for each [i] rather than globally otherwise the
        # correct frequency order gets lost.
        z_interp[i] = np.flip(z_interp[i])  # reverse z_interp
        r_interp[i] = np.flip(r_interp[i])  # reverse r_interp
        kz_ext[i] = np.flip(kz_ext[i])  # reverse kz_ext
        kz_array[i] = np.flip(kz_array[i])  # reverse kz_array
        fz_array[i] = np.flip(fz_array[i])

        if prm.flux_debug_mode:
            print(f"z_interp[{i}] \n {z_interp[i]}")
            print(f"fz_array[{i}] \n {fz_array[i]}")
            print(f"kz_array[{i}] \n {kz_array[i]}")

    if freq_dict_on_z:
        return kz_ext, kz_array, fz_array, z_interp, r_interp
    else:
        print(f"freq_dict_on_z empty! Abort program.")
        exit()


# -----------------------------------------------------------------
def compute_grid_quantities(idp, z_interp, r_interp, source, taur):
    # -------------------------------------------------------------
    """
    computes all relevant physical quantities on z grid:
    tau_interp, source_interp, deltau_r
    """
    print("Entering compute_grid_quantities") if prm.flux_debug_mode else None
    # r_interp = np.zeros((idp), object)
    # assign line-dependent arrays
    source_interp = np.zeros((idp, prm.nline), object)
    deltau_r = np.zeros((idp, prm.nline), object)

    for i in range(idp):
        if prm.flux_debug_mode:
            print(f"\nfrom compute_grid_quantities - i = {i}")
            print(f"z_interp[{i}] \n{z_interp[i]}")
        for lx in range(prm.nline):
            tau_interp = np.interp(r_interp[i], grid.r, taur[lx, :])
            source_interp[i, lx] = np.interp(r_interp[i], grid.r, source[lx, :])
            deltau_r[i, lx] = tau_interp / lib.der_z_num(r_interp[i], z_interp[i])
            # ---------------------------------------------------------------------------------
            # CAUTION: following line valid only for power law velocity fields - see grid.v_law
            # for the general case, use above line instead
            # deltau_r[i, lx] = tau_interp / lib.der_z(r_interp[i], z_interp[i])
            # ---------------------------------------------------------------------------------
            if prm.flux_debug_mode:
                print(f"--- i = {i}, lx = {lx}  "
                      f"tau_interp \n {tau_interp}")
                # print(f"s_interp \n {source_interp[i, lx]}")
                # print(f"deltau_r \n {deltau_r[i, lx]}")

    return source_interp, deltau_r


# -----------------------
def vrad(idp, rzpj, zpj):
    # -------------------
    # compute radial velocities for all z grid points on all impact parameters
    #
    vrmin = np.zeros(idp)
    vrmax = np.zeros(idp)

    vr = [lib.vrad_z(rzpj[i], zpj[i]) for i in range(idp)]

    for i in range(idp):
        vrmin[i] = np.min(vr[i])
        vrmax[i] = np.max(vr[i])

        if prm.flux_debug_mode:
            print(f"vr[{i}] \n {vr[i]}")
            print(f"vrmin[{i}] = {vrmin[i]:.2f}, vrmax[{i}] = {vrmax[i]:.2f}")

    return vr, vrmin, vrmax


# ----------------------------
def add_line_wings(nwing, kz):
    # ------------------------
    """
    add line wings witn nwing points to both sides of the kz vector
    and return the extended array kzl
    ordering depends on the slope of first two and last two kz points
    """
    if len(kz) > 1:
        kzl = kz
        slope0 = kz[1] - kz[0]
        if prm.flux_debug_mode:
            if slope0 > 0:
                print(f"v[1] = {kz[1]}, v[0] = {kz[0]}, slope0 > 0")
            else:
                print(f"v[1] = {kz[1]}, v[0] = {kz[0]}, slope0 <= 0")

        slope1 = kz[-1] - kz[-2]
        if prm.flux_debug_mode:
            if slope1 > 0:
                print(f"v[-2] = {kz[-2]}, v[-1] = {kz[-1]}, slope1 > 0")
            else:
                print(f"v[-2] = {kz[-2]}, v[-1] = {kz[-1]}, slope1 <= 0")

        if slope0 > 0:
            kz0 = kz[0] - nwing
            kz1 = kz[0]
        else:
            kz1 = kz[0] + nwing + 1
            kz0 = kz[0] + 1
        kzm = np.array([j for j in range(kz0, kz1)], int)
        print(f"kzm = {kzm}") if prm.flux_debug_mode else None
        if slope0 <= 0:
            kzm = kzm[::-1]
        kzl = np.append(kzm, kzl)

        if slope1 > 0:
            kz1 = kz[-1] + nwing
            kz0 = kz[-1] + 1
        else:
            kz0 = kz[-1] - nwing
            kz1 = kz[-1]
        kzp = np.array([j for j in range(kz0, kz1)], int)
        print(f"kzp = {kzp}") if prm.flux_debug_mode else None
        if slope1 <= 0:
            kzp = kzp[::-1]
        kzl = np.append(kzl, kzp)
        print(f"extended kzl = {kzl}") if prm.flux_debug_mode else None

        return kzl
    else:
        return kz


# -----------------------------------------------------------
def adjust_kz_vel(idp, nwing, fz, kz_vel, delta_v_mat, dvel):
    # -------------------------------------------------------
    """
    here we introduce the frequency difference delta_f between lines
    all frequencies on p will contribute to the bluest line (in accelerated
    outflows) but the contribution from the bluest line to redder lines comes
    from a smaller frequency interval due to the Doppler shift. For each line,
    we need to find the reduced frequency range fz_l.
    """
    fz_l = np.zeros((idp, prm.nline, prm.nline), object)
    kz_l = np.zeros((idp, prm.nline, prm.nline), object)
    k_l = np.zeros((idp, prm.nline, prm.nline), object)

    print(f'\nEntering adjust_kz_vel') if prm.flux_debug_mode else None
    # fz is the frequency grid along the ray i
    # kz_vel is the frequency index along ray i
    # both are line-independent
    for i in range(idp):
        if prm.flux_debug_mode:
            print(f"\n ----- fz[{i}] \n{fz[i]}")
        # mono_flag = lib.monotonic(fz[i])
        # print(f"fz[{i}] monotonic? {mono_flag}") if prm.flux_debug_mode else None
        fz_max = np.max(fz[i])
        fz_min = np.min(fz[i])
        fz_0 = fz[i][0]
        fz_1 = fz[i][-1]
        print(f"\nfz_0 = {fz_0}, fz_1 = {fz_1}, fz_min = {fz_min}, fz_max = {fz_max}\n") \
            if prm.flux_debug_mode else None
        print(f"kz_vel[{i}] \n{kz_vel[i]}") if prm.flux_debug_mode else None

        for lx in range(prm.nline):
            # loop over all lines that interact with lx
            for mx in [x for x in range(prm.nline) if x != lx]:
                if prm.flux_debug_mode:
                    print('----------------------------------------------------------------------------------------')
                    print(f"i = {i}, lx = {lx}, mx = {mx}, delta_v_mat[{lx, mx}] = {delta_v_mat[lx, mx]}")
                    print('----------------------------------------------------------------------------------------')
                # ------------------------------------------------------------
                # fz_l is the frequency grid adjusted for the frequency
                # displacement between line mx and line lx
                # ------------------------------------------------------------------
                fz_l[i, lx, mx] = np.array([f - delta_v_mat[lx, mx] for f in fz[i]])
                # ------------------------------------------------------------------
                print(f"fz_l[{i},{lx},{mx}] \n {fz_l[i, lx, mx]}") if prm.flux_debug_mode else None

                # -----------------------------------------------------------------------------------------
                # because of the Doppler shift, fz_l involves frequencies that are not within the fz limits
                # so we must now adjust fz_l to stay within the fz boundaries.
                # the part of fz that will be deleted depends on the nature of the flow:
                # if the considered line is redder, then fz_l extends beyond the reddest frequency in fz;
                # this corresponds to an accelerated flow according to our line ordering convention
                # if the considered line is bluer, then fz_l extends beyond the bluest frequency in fz;
                # this corresponds to the decelerated flow case
                # now remember that the frequencies are ordered from blue to red on the ray, and find
                # the indices of the part that extends beyond the fz limit, then delete frequencies
                # that are not present on the considered impact parameter
                # -------------------------------------------------------------------------------------------
                fz_l[i, lx, mx] = fz_l[i, lx, mx][np.in1d(fz_l[i, lx, mx], fz[i])]

                if prm.flux_debug_mode:
                    print(f"shortened fz_l[{i},{lx},{mx}] \n {fz_l[i, lx, mx]}")
                    print(f"shape(fz_l) = {np.shape(fz_l)}, len(fz_l) = {len(fz_l[i, lx, mx])}")

                # same procedure for frequency index
                # ------------------------------------------------------------------------
                kz_l[i, lx, mx] = np.array([k - delta_v_mat[lx, mx] / dvel for k in kz_vel[i]], int)
                # ------------------------------------------------------------------------
                print(f"kz_l[{i},{lx},{mx}] \n {kz_l[i, lx, mx]}") if prm.flux_debug_mode else None

                kz_l[i, lx, mx] = kz_l[i, lx, mx][np.in1d(kz_l[i, lx, mx], kz_vel[i])]
                if prm.flux_debug_mode:
                    print(f"shortened kz_l[{i},{lx},{mx}] \n {kz_l[i, lx, mx]}")
                    print(f"shape(kz_l) = {np.shape(kz_l)}, len(kz_l) = {len(kz_l[i, lx, mx])}")

                # kl is the equivalent of kz, it is extended to include nwing on both sides
                # this is done in function add_line_wings
                if kz_l[i, lx, mx].any():
                    k_l[i, lx, mx] = add_line_wings(nwing, kz_l[i, lx, mx])
                    print(f"extended k_l[{i},{lx},{mx}] \n {k_l[i, lx, mx]}") \
                        if prm.flux_debug_mode else None

    return fz_l, kz_l, k_l


# -------------------------------------------
def find_interactions(idp, kz_vel, kz_vel_l):
    # ---------------------------------------
    """
    look for interactions between lines, which occur whenever the kz_vel's of the different lines intersect.
    kz_vel_intersect holds the interacting frequencies while p_couplings records which lines interact on a given ray
    """
    kz_vel_intersect = np.zeros((idp, prm.nline, prm.nline), object)
    p_couplings = np.zeros((idp, prm.nline), object)
    print(f"Entering find_interactions") if prm.flux_debug_mode else None
    for i in range(idp):
        for lx in range(prm.nline):
            print(f"----------------------i = {i}  lx = {lx}") if prm.flux_debug_mode else None
            print(f"kz_vel[{i}] \n{kz_vel[i]}") if prm.flux_debug_mode else None
            p_coupling = []

            for mx in [x for x in range(prm.nline) if x != lx]:
                print(f"---------------------------  mx = {mx}") if prm.flux_debug_mode else None
                print(f"kz_vel_l[{i},{lx},{mx}] \n{kz_vel_l[i, lx, mx]}") if prm.flux_debug_mode else None

                kz_vel_intersect[i, lx, mx] = kz_vel[i][np.in1d(kz_vel[i], kz_vel_l[i, lx, mx])]

                print(f"\nkz_vel_intersect[{i},{lx},{mx}] \n{kz_vel_intersect[i, lx, mx]}") \
                    if prm.flux_debug_mode else None
                # caution: np.intersect1d returns a sorted array, so we need to reverse it
                # for consistency with order of kz_vel and z_interp
                # kz_vel_intersect[i, lx, mx] = np.flip(kz_vel_intersect[i, lx, mx])

                if np.any(kz_vel_intersect[i, lx, mx]):
                    print(f"interaction on ray i = {i} for lines lx = {lx}, m = {mx}") \
                        if prm.flux_debug_mode else None
                    p_coupling.append(mx)

            p_couplings[i, lx] = p_coupling
            print(f"from find_interactions - p_couplings[{i},{lx}] = {p_coupling}") if prm.flux_debug_mode else None

    print(f"p_couplings \n{p_couplings}") if prm.flux_debug_mode else None
    print(f"exit find_interactions") if prm.flux_debug_mode else None

    return kz_vel_intersect, p_couplings


# --------------------------------------------------------------------------------------------------------------
def dint_loop(ip, iz, lx, index, p_couplings, nfreq, nwing, dxd, kz_vel_intersect, source_interp, deltau_r, dint,
              exptau, exptau_tot):
    # ----------------------------------------------------------------------------------------------------------
    """
    This is the innermost loop of the flux integration for line lx.
    It computes the incremental intensity on impact parameter ip and at kz_vel location iz,
    while allowing for a possible interaction with 1 or 2 other computed lines
    """
    exp_deltau_l = np.ones(prm.nline)
    dint_l = np.zeros(prm.nline)
    if prm.flux_debug_mode:
        print(f"entering dint_loop with ip = {ip}, lx = {lx}, p_couplings = {p_couplings[ip, lx]}")

    # j runs over the local absorption line profile
    j_range = [j for j in range(2 * nwing)] if nwing else [0]

    for j in j_range:

        # first compute the intensity increment due to lx photons
        k = nfreq + 1 - index - nwing + j
        print(f"index = {index}, iz = {iz}, j = {j}, k = {k}") if prm.flux_debug_mode else None
        exp_deltau_l[lx] = np.exp(-deltau_r[ip, lx][iz] * dxd[lx, j])
        dint_l[lx] = source_interp[ip, lx][iz] * (1.0 - exp_deltau_l[lx]) * exptau[lx, k]
        dint[lx, k] += dint_l[lx]
        exptau[lx, k] *= exp_deltau_l[lx]
        exptau_tot[lx, k] = exptau[lx, k]

        # then compute the intensity increment(s) due to photons of interacting line(s) mx or mx and nx
        if len(p_couplings[ip, lx]) == 1:
            print(f"entering dint_loop with lx = {lx}, mx = {p_couplings[ip, lx][0]}") if prm.flux_debug_mode else None
            mx = p_couplings[ip, lx][0]

            if index in kz_vel_intersect[ip, lx, mx]:
                exp_deltau_l[mx] = np.exp(-deltau_r[ip, mx][iz] * dxd[mx, j])
                dint_l[mx] = source_interp[ip, mx][iz] * (1.0 - exp_deltau_l[mx]) \
                    * exptau[mx, k] * exptau[lx, k]
                dint[lx, k] += dint_l[lx] + dint_l[mx]
                exptau[mx, k] *= exp_deltau_l[mx]
                exptau_tot[lx, k] = exptau[lx, k] * exptau[mx, k]

        elif len(p_couplings[ip, lx]) == 2:
            print(f"entering dint_loop with lx = {lx}, mx = {p_couplings[ip, lx][0]}, nx = {p_couplings[ip, lx][1]}'") \
                if prm.flux_debug_mode else None
            mx = p_couplings[ip, lx][0]
            nx = p_couplings[ip, lx][1]
            if index in kz_vel_intersect[ip, lx, mx] and index not in kz_vel_intersect[ip, lx, nx]:
                exp_deltau_l[mx] = np.exp(-deltau_r[ip, mx][iz] * dxd[mx, j])
                dint_l[mx] = source_interp[ip, mx][iz] * (1.0 - exp_deltau_l[mx]) \
                    * exptau[mx, k] * exptau[lx, k]
                dint[lx, k] += dint_l[lx] + dint_l[mx]
                exptau[mx, k] *= exp_deltau_l[mx]
                exptau_tot[lx, k] = exptau[lx, k] * exptau[mx, k]

            elif index in kz_vel_intersect[ip, lx, nx] and index not in kz_vel_intersect[ip, lx, mx]:
                exp_deltau_l[nx] = np.exp(-deltau_r[ip, nx][iz] * dxd[nx, j])
                dint_l[nx] = source_interp[ip, nx][iz] * (1.0 - exp_deltau_l[nx]) \
                    * exptau[nx, k] * exptau[mx, k] * exptau[lx, k]
                dint[lx, k] += dint_l[lx] + dint_l[nx]
                exptau[nx, k] *= exp_deltau_l[nx]
                exptau_tot[lx, k] = exptau[lx, k] * exptau[nx, k]

            elif index in kz_vel_intersect[ip, lx, mx] and index in kz_vel_intersect[ip, lx, nx]:
                exp_deltau_l[mx] = np.exp(-deltau_r[ip, mx][iz] * dxd[mx, j])
                exp_deltau_l[nx] = np.exp(-deltau_r[ip, nx][iz] * dxd[nx, j])
                dint_l[mx] = source_interp[ip, mx][iz] * (1.0 - exp_deltau_l[mx]) \
                    * exptau[mx, k] * exptau[lx, k]
                dint_l[nx] = source_interp[ip, nx][iz] * (1.0 - exp_deltau_l[nx]) \
                    * exptau[nx, k] * exptau[mx, k] * exptau[lx, k]
                dint[lx, k] += dint_l[lx] + dint_l[mx] + dint_l[nx]
                exptau[mx, k] *= exp_deltau_l[mx]
                exptau[nx, k] *= exp_deltau_l[nx]
                exptau_tot[lx, k] = exptau[lx, k] * exptau[mx, k] * exptau[nx, k]

            elif index not in kz_vel_intersect[ip, lx, mx] and index not in kz_vel_intersect[ip, lx, nx]:
                print(f"entering dint_loop without interaction") if prm.flux_debug_mode else None
                pass

        elif len(p_couplings[ip, lx]) > 2:
            print(f"from dint_loop - len(p_couplings[{ip}, {lx}]) = {len(p_couplings[ip, lx])})"
                  f" - unforeseen case? abort program")
            exit()

        else:
            # no interaction on this ray, the only photons contributing
            # to the intensity are those of lx itself
            pass

    return dint, exptau, exptau_tot


# -----------------------------------------------------------------------------------------------
def emer(region, ip, lx, p_couplings, nfreq, nfreq_tot, nwing, dxd, kz, kz_vel, kz_vel_intersect,
         source_interp, deltau_r, fc):
    # -------------------------------------------------------------------------------------------
    """
    computes the emergent intensities on a line of sight for all lines by summing
    S(1-exp(-dtau*dphi))*exp(-tau) and adding the core contribution.
    at each z corresponding to some line displacement kz_vel, the line integration
    runs over the local profile function, which extends over (0, 2*prm.nwing),
    so in absence of interactions we have non-zero intensity contributions to frequencies
    in the interval (kz_vel-prm.nwing, kz_vel+prm.nwing)
    CAUTION: kz_vel must be in ascending order in an accelerated outflow, but reversed in a
    decelerated one. In principle, it does that automatically. Kz can be in either ascending
    or descending order. MORE CAUTION: physical quantities should be ordered from zmax to zmin
    in all cases.
    """
    dint = np.zeros((prm.nline, nfreq_tot))
    exptau = np.ones((prm.nline, nfreq_tot))
    exptau_tot = np.ones((prm.nline, nfreq_tot))

    print(f"\nEntering emer - ip = {ip}, lx = {lx}, nfreq = {nfreq}, nwing = {nwing}") \
        if prm.flux_debug_mode else None

    iz = -1
    if prm.flux_debug_mode:
        print(f"kz_vel \n{kz_vel}")  # if prm.flux_debug_mode else None
        print(f"source_interp[{ip},{lx}] \n{source_interp[ip, lx]}")
        print(f"deltau_r[{ip},{lx}] \n{deltau_r[ip, lx]}")

    # for all velocity displacements found on the ray...
    for index in kz_vel[ip]:
        iz += 1
        # compute contributions to the intensity for all frequencies present in the local profile
        # print(f"calling dint_loop: ip, iz, index, lx = {ip, iz, index, lx}, dint[lx, index] = {dint[lx, index]}")
        dint_loop(ip, iz, lx, index, p_couplings, nfreq, nwing, dxd, kz_vel_intersect, source_interp, deltau_r,
                  dint, exptau, exptau_tot)

    # then add the contribution from attenuated core continuum
    # to the intensity of rays originating from it
    print(f"from emer with ip = {ip}, lx = {lx}") if prm.flux_debug_mode else None
    if region == 'core':
        dinc = np.ones((prm.nline, nfreq_tot)) * fc[lx]
        print(f"kz \n{kz}") if prm.flux_debug_mode else None
        for index in kz[ip]:
            k = nfreq + 1 - index
            dinc[lx, k] *= exptau[lx, k]

        for j in range(nfreq_tot):
            dint[lx, j] += dinc[lx, j]

        print(f"dinc[{lx}, :] \n{dinc[lx, :]}") if prm.flux_debug_mode else None
        print(f"dint[{lx}, :] \n{dint[lx, :]}") if prm.flux_debug_mode else None

    else:
        print(f"dint[{lx}, :] \n{dint[lx, :]}") if prm.flux_debug_mode else None

    return dint[lx, :]


# -------------------------------------------------------------------------------------
def integrate_flux(region, idp, nfreq, nfreq_tot, nwing, dxd, f_dict, delta_v_mat, dvel,
                   pj, wj, source, taur, fc):
    # ----------------------------------------------------------------------------------
    """
    compute the line flux over a set of impact parameters pj,
    using a Gaussian numerical integration
    vr, fz, kz_vel, z_interp and r_interp are envelope properties, thus line-independent
    """
    emer_intensity = np.zeros((prm.nline, nfreq_tot))
    flux = np.zeros((prm.nline, nfreq_tot))

    # initialize p_couplings and kv_vel_intersect for non-interacting case
    kz_vel_intersect = np.zeros((idp, prm.nline, prm.nline), object)
    p_couplings = np.zeros((idp, prm.nline), object)
    for i in range(idp):
        for lx in range(prm.nline):
            p_couplings[i, lx] = []
            for m in range(prm.nline):
                kz_vel_intersect[i, lx, m] = []

    # define z-grid over impact parameters p
    zpj, rzpj = z_grid(region, idp, pj)

    # compute radial velocities for all p,z
    vr, vrmin, vrmax = vrad(idp, rzpj, zpj)

    # find all line frequencies present on all impact parameters and return their z location
    # all quantities are computed in the stellar rest frame
    kz, kz_vel, fz, z_interp, r_interp = freq_grid(idp, pj, zpj, vr, f_dict, nwing, vrmin, vrmax)

    # compute the relevant physical properties on all impact parameters and all z's
    source_interp, deltau_r = compute_grid_quantities(idp, z_interp, r_interp, source, taur)

    if prm.interact_mode:
        # find the contributions from the blue line(s) to redder ones
        # currently, only accelerating outflows and inflows are considered in this function
        fz_l, kz_vel_l, kz_l = adjust_kz_vel(idp, nwing, fz, kz_vel, delta_v_mat, dvel)

        # find which lines interact with any given one.
        kz_vel_intersect, p_couplings = find_interactions(idp, kz_vel, kz_vel_l)

    # compute the emergent intensity on each ray, integrate it along the impact parameters and get the flux
    for lx in range(prm.nline):
        emer_intensity[lx] = sum([emer(region, ip, lx, p_couplings, nfreq, nfreq_tot, nwing,
                                       dxd, kz, kz_vel, kz_vel_intersect, source_interp,
                                       deltau_r, fc) * wj[ip] * pj[ip] for ip in range(idp)])

        # finally, normalize the gaussian integral and return the flux
        flux[lx, :] = (np.abs(pj[-1] - pj[0])) * emer_intensity[lx, :]

    return flux
