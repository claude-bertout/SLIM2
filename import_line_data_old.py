"""
Module name: import_line_data (20/22/2023 version with i_vmax)
This module imports the line list and their atomic data from import_atomic_data.
Lines are imported in order of increasing atomic weight and reordered according
to the nature of the velocity field after the radial optical depths are computed.
Matrices of possible interactions betweenlines are set up for use during the
source function computation. The module also computes taur, the LTE line center tangential
optical depth vector for all lines, using the LTE_module.
Furthermore, the line absortion profiles are defined here
as well as all parameters of the frequency grid for the flux integration, including a
common velocity frame that is used for all the considered lines.
Finally, we set up the dictionary of all frequencies/velocities  that are included in the line computation.
NB. The exported taur vector includes the normalization factor 1/local_profile_width. The localization of this
normalization needs to be moved to integrate_flux if the envelope temperature is not constant.

"""
import numpy as np
import scipy.special as spec
import scipy.interpolate as interpol
import import_parameters as prm
import import_atomic_data as adat
import set_grids as grid
import constants as const
import LTE_module as lte
import plot_library as plres

global atom, ion_level, a_weight, abund, lambda0, El_eV, Eu_eV, gl, gu, flu, Aul, log_gf, ion_level_l, ion_level_u


# atomic line broadening
# ------------------------------------
def broadening(atomic_weight, temp0, vturb0):
    # --------------------------------
    """
    Returns the local absorption line broadening width (in units of velocity) of an atomic line
    with atomic weight A, at temperature T and turbulent broadening vturb
    Divide by lambda0 to get Doppler width in frequency units.
    Checked vs. Gray p.249 except constant np.sqrt(2.0*k_B/m_p)/cvel
    is 4.286e-7 here instead of 4.301e-7 in Gray
    """
    return np.sqrt(2.0 * const.k_B * temp0 / (atomic_weight * const.m_p) + vturb0 ** 2)


# ---------------------------------------------------------
def variable_local_profile_integral(nwing, nwingl, dfreql):
    # -----------------------------------------------------
    """
    computes the local profile function integral dxd
    the profile itself is assumed to be Gaussian
    this version handles variable profile widths (when lines
    of different elements are computed).
    """
    if nwing:
        dxd = np.zeros(nwing)
        xdminus = [dfreql * j for j in range(nwingl)]
        xdplus = [dfreql * (j + 1) for j in range(nwingl)]
        dxd2 = [(spec.erf(xdplus[j]) - spec.erf(xdminus[j])) / 2.0 for j in range(nwingl)]
        dxd[0:nwingl] = dxd2
        dxd1 = dxd[::-1]
        dxd = np.append(dxd1, dxd)
    else:
        dxd = np.zeros(1)
        dxd[0] = 1.0

    return dxd


# -----------------------------------------
def collision_de_exc_rate(wavel, temp, ne):
    # -------------------------------------
    # approximates the lines collisional de-excitation rates by interpolating
    # Van Regemorter's (1962) table for the P integral

    global atom, ion_level, a_weight, abund, lambda0, El_eV, Eu_eV, gl, gu, flu, Aul, log_gf, ion_level_l, ion_level_u

    # Data: Van Regemorter's table
    e_kt = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10])  # abscissa (DeltaE/kT)
    p_neutrals = np.array([1.16, 0.96, 0.70, 0.49, 0.33, 0.17, 0.1, 0.063, 0.035, 0.023])  # P integral for neutrals
    p_ions = np.array([1.16, 0.98, 0.74, 0.55, 0.40, 0.26, 0.22, 0.21, 0.20, 0.20])  # P integral for ions

    coll_rate = np.zeros((prm.nline, prm.idr))
    de_kt = np.zeros((prm.nline, prm.idr))

    delta_e = Eu_eV - El_eV
    for lx in range(prm.nline):
        de_kt[lx, :] = [delta_e[lx] / (const.k * t) for t in temp]

    dmin = np.where(de_kt < e_kt[0], 0, de_kt)
    dmax = np.where(de_kt > e_kt[-1], 0, de_kt)
    if not dmin.any() or not dmax.any():
        print(f" From ld.collision_de_exc_rate: de_kt not in e_kt range. Abort")
        exit()
    p_integral = np.zeros(prm.idr)
    for lx in range(prm.nline):
        if ion_level_l[lx] > 0.0:  # ion
            p_coll = p_ions
        else:  # neutral
            p_coll = p_neutrals
        # p_integral = interpol.interp1d(e_kt, p_coll, kind='linear')
        p_integral[:] = np.interp(de_kt[lx, :], e_kt, p_coll)
        # for i in range(prm.idr):
            # dinter, iend = interpol_cb(1, 9, e_kt, de_kt[lx, i])
            # p_integral = p_coll[int(iend) - 1] + dinter * (p_coll[int(iend)] - p_coll[int(iend) - 1])
            # coll_rate[lx, i] = 2.0e1 * ne[i] * grid.nt[i] * p_integral(de_kt[lx, i]) * wavel[lx] ** 3 / np.sqrt(temp[i])
        coll_rate[lx, :] = 2.0e1 * ne[:] * grid.nt[:] * p_integral[:] * wavel[lx] ** 3 / np.sqrt(temp[:])
        print(f"coll_rate[{lx}, :] \n{coll_rate[lx, :]}") if prm.debug_mode else None

    return coll_rate


# ---------------------------------------
def reorder_lines(line_properties, taur):
    # -----------------------------------
    """
    reorder lines in order of increasing or decreasing wavelengths
    depending on the velocity field.
    NB. This reordering is not strictly necessary for the computation
    but useful for the plotting methods
    """
    line_properties = np.array(line_properties)

    print(f"\nOrdering lines from blue to red")
    # remember that lambda0[lx] = line_properties[lx, 4]
    line_properties_sorted = line_properties[line_properties[:, 4].argsort()]
    taur_sorted = taur[line_properties[:, 4].argsort()]

    # reverse the line order in case of infall (reddest line first)
    if prm.velocity_index == 1 or prm.velocity_index == 3:
        print(f"Reordering lines and optical depths from red to blue\n")
        line_properties_reordered = line_properties_sorted[::-1]
        taur_reordered = taur_sorted[::-1]
    else:
        line_properties_reordered = line_properties_sorted
        taur_reordered = taur_sorted

    for lx in range(prm.nline):
        atom[lx], ion_level[lx], a_weight[lx], abund[lx], lambda0[lx], El_eV[lx], Eu_eV[lx], gl[lx], gu[lx], \
            Aul[lx], flu[lx], log_gf[lx], ion_level_l[lx], ion_level_u[lx] = line_properties_reordered[lx]
        if prm.debug_mode:
            print(f"\nLine_properties for {lx} after reordering \n{line_properties[lx]}")
            print(f"radial optical depth after reordering for {lx} \n{taur_reordered[lx]}")

    return line_properties_reordered, taur_reordered


# -----------------------------------
def compute_optical_depths(nt, temp):
    # -------------------------------
    """
    computes the radial part of the line optical depths
    :param nt: np.array(prm.idr)
    :param temp: np.array(prm.idr)
    :return: lambda0: np.array(prm.nline), mdot: float, line_d_width: np.array((prm.nline, prm.idr))
    taur: np.array((prm.nline, prm.idr))
    """
    global atom, ion_level, a_weight, abund, lambda0, El_eV, Eu_eV, gl, gu, flu, Aul, log_gf, ion_level_l, ion_level_u

    atom = np.empty(prm.nline, str)
    ion_level = np.zeros(prm.nline)
    a_weight = np.zeros(prm.nline)
    abund = np.zeros(prm.nline)
    lambda0 = np.zeros(prm.nline)
    El_eV = np.zeros(prm.nline)
    Eu_eV = np.zeros(prm.nline)
    gl = np.zeros(prm.nline)
    gu = np.zeros(prm.nline)
    flu = np.zeros(prm.nline)
    Aul = np.zeros(prm.nline)
    log_gf = np.zeros(prm.nline)
    ion_level_l = np.zeros(prm.nline)
    ion_level_u = np.zeros(prm.nline)

    # the list line_properties is used to accumulate atomic data for lines under consideration
    line_properties = []
    for index, element in enumerate(prm.elements):
        print(f"\nindex = {index}, element = {element}")  # if prm.debug_mode else None
        adat.read_line_data(element[0], element[1], prm.line_range[index], line_properties)

    print(f"return from read_line_data")  # if prm.debug_mode else None
    for lx in range(prm.nline):
        atom[lx], ion_level[lx], a_weight[lx], abund[lx], lambda0[lx], El_eV[lx], Eu_eV[lx], gl[lx], gu[lx], \
            Aul[lx], flu[lx], log_gf[lx], ion_level_l[lx], ion_level_u[lx] = line_properties[lx]

    # taur = np.zeros((prm.nline, prm.idr))
    # compute lte optical depths for the lines whose properties are passed as global variables
    # CAUTION: the elements must be sorted by atomic weight to compute taur
    ne, nl, nu, mu, taur = lte.lte_tau(atom, ion_level, abund, lambda0, gl, gu, El_eV, Eu_eV, flu, nt, temp)

    if prm.debug_mode:
        for lx in range(prm.nline):
            print(f"\nline center optical depth for lx = {lx} before reordering\n{taur[lx, :]}")
        print(f"Shape taur: {np.shape(taur)}")

    print(f"\nnt[0] = {nt[0]:.2e} cm^-3, v[0] = {grid.v_env[0]:.2e} km/s, dmu = {np.round(mu[0], 2)}\n")

    line_properties_reordered, taur_reordered = reorder_lines(line_properties, taur)
    line_d_width = np.zeros(prm.nline)

    for lx in range(prm.nline):
        lambda0[lx] = line_properties_reordered[lx, 4]
        a_weight[lx] = line_properties_reordered[lx, 2]
        line_d_width[lx] = broadening(a_weight[lx], temp[0], prm.vturb0) * const.km_s
    for lx in range(prm.nline):
        print(f"element = {line_properties_reordered[lx][0]}, lambda0[{lx}] = {lambda0[lx] / const.angstrom:.2f}A, "
              f"a_weight[{lx}] = {a_weight[lx]}, line_d_width[{lx}] = {line_d_width[lx]:.2f} km/s")

    taur_normalized = np.zeros((prm.nline, prm.idr))
    # include the profile width normalization in taur
    for lx in range(prm.nline):
        taur_normalized[lx, :] = taur_reordered[lx, :] / (line_d_width[lx] / const.km_s) / lambda0[lx]
        print(f"\nNormalized taur vector for l = {lx} after reordering \n{taur_normalized[lx, :]}") if prm.debug_mode else None

    return lambda0, line_d_width, ne, taur_normalized


# ------------------------------------
def compute_interaction_matrix(wavel):
    # --------------------------------
    """
    delta_v is the vector of line displacements with respect to reference line 0.
    delta_olson is the vector of velocity displacement normalized to the maximum envelope velocity.
    Named after Gordon L. Olson, who first introduced the criterion delta_olson < 1 for line interaction.
    NB. The choice of a reference line is in fact arbitrary, and chosen here to make sure that the
    plotted lines inherit the right wavelengths and names.
    delta_v_mat is the matrix of velocity displacements between all lines.
    int_mat is a boolean matrix telling which lines are part of the blend of interacting lines
    based on the velocity shift between lines.
    Both delta_v_mat and int_mat are valid for an arbitrary number of lines.
    self_cp and inter_cp are boolean matrices keeping track of which cp surfaces must be computed
    for which lines.
    NB. convention: negative toward red
    """

    delta_v = np.zeros(prm.nline)
    delta_olson = np.zeros(prm.nline)
    vmax = max(grid.v_law(grid.r))

    for lx in range(1, prm.nline):
        delta_wav = wavel[0] - wavel[lx]
        delta_v[lx] = float(np.around(const.cvel * const.km_s * delta_wav / wavel[0]))  # in km/s
        delta_olson[lx] = np.abs(delta_wav / wavel[0] * const.cvel / (vmax * 1.0e5))
        if prm.debug_mode:
            print(f"delta_v[{lx}] relative to line 0 = {delta_v[lx]: .2f} km / s, delta_olson = {delta_olson[lx]:.2f}")

    # ------------------------------------------------------------------
    # delta_v_mat is the matrix of velocity displacements between lines,
    # each of them being in turn the reference line
    # -------------------------------------------------------------------
    delta_v_mat = np.zeros((prm.nline, prm.nline))
    delta_olson_mat = np.zeros((prm.nline, prm.nline))
    # int_mat = np.zeros((prm.nline, prm.nline), bool)
    inter_cp = np.zeros((prm.nline, prm.nline))
    self_cp = np.zeros((prm.nline, prm.nline))

    delta_v_mat[0, :] = delta_v[:]
    for j in range(1, prm.nline):
        delta_v_mat[j, :] = [delta_v_mat[j - 1, k] - delta_v_mat[0, j] + delta_v_mat[0, j - 1] for k in
                             range(prm.nline)]
        if prm.debug_mode:
            print(f"\ndelta_v_mat relative to line 0\n{delta_v_mat}\n") if prm.debug_mode else None
    for j in range(prm.nline):
        delta_olson_mat[j, :] = [np.round(np.abs(delta_v_mat[j, k] / vmax), 2) for k in range(prm.nline)]
    print(f"\ndelta_olson_mat\n{delta_olson_mat}\n") if prm.debug_mode else None

    # Boolean matrix int_mat identifies the interacting lines (delta_Olson <1))
    # in the overall delta_olson_mat matrix
    int_mat = np.where(delta_olson_mat < 1.0, delta_olson_mat * vmax, 0)
    print(f"\nint_mat\n{int_mat}\n") if prm.debug_mode else None

    # int_line is the set of interacting lines
    int_lines = []
    for lx in range(prm.nline):
        for lxx in [x for x in range(prm.nline) if x != lx]:
            if int_mat[lx, lxx]:
                int_lines.append(int(lxx))
    int_lines = sorted(set(int_lines))
    print(f"interacting lines int_lines = {int_lines}")
    for lx in range(prm.nline):
        delta_wav = wavel[int_lines[0]] - wavel[lx]
        delta_v[lx] = float(np.around(const.cvel * const.km_s * delta_wav / wavel[int_lines[0]]))  # in km/s
    if prm.debug_mode:
        print(f"\ndelta_v relative to line {int_lines[0]} = {delta_v} km / s")
    delta_v_mat = np.zeros((prm.nline, prm.nline))
    delta_v_mat[int_lines[0], :] = delta_v[:]
    for j in range(prm.nline):
        if j == int_lines[0]:
            pass
        else:
            delta_v_mat[j, :] = [delta_v_mat[j - 1, k] - delta_v_mat[int_lines[0], j] + delta_v_mat[int_lines[0], j - 1]
                                 for k in range(prm.nline)]
    print(f"\ndelta_v_mat relative to line {int_lines[0]}\n{delta_v_mat}\n") if prm.debug_mode else None

    int_vmax = abs(delta_v[int_lines[-1]] - delta_v[int_lines[0]])
    print(f"maximum displacement between interacting lines i_vmax = {int_vmax} km/s") if prm.debug_mode else None

    # ---------------------------------------------------------------------------------------
    # look for possible interactions (giving rise to CP surfaces) between the different lines
    # ---------------------------------------------------------------------------------------

    if prm.interact_mode or prm.non_local:
        print(f"\nInteractions between computed lines")
        print(f"non_local = {prm.non_local}, interact_mode = {prm.interact_mode}")
        for lx in range(prm.nline):
            print(f"line lx = {lx}")
            for lxx in range(prm.nline):
                if lx == lxx and (grid.accel_flag == 'False' or grid.accel_flag == 'Both'):
                    print(f"lx = {lx}, lxx = {lxx} look for own CP surface")
                    self_cp[lx, lxx] = 1
        for lx in int_lines:
            print(f"line lx = {lx}")
            for lxx in reversed(int_lines):
                if int_vmax >= delta_v_mat[lx, lxx] > 0.0 and (grid.accel_flag == 'True' or grid.accel_flag == 'Both') \
                        and prm.interact_mode:
                    inter_cp[lx, lxx] = 1 if np.abs(delta_v_mat[lx, lxx]) < vmax else 0
                    print(f"lx = {lx}, lxx = {lxx} look for red neighbor CP")
                elif - int_vmax <= delta_v_mat[lx, lxx] < 0.0 and (grid.accel_flag == 'False'
                                                                   or grid.accel_flag == 'Both') and prm.interact_mode:
                    inter_cp[lx, lxx] = 1
                    print(f"lx = {lx}, lxx = {lxx} look for blue neighbor CP")
                else:
                    print(f"lx = {lx}, lxx = {lxx} -- no interaction")
        print(f"self_cp\n{self_cp}\ninter_cp\n{inter_cp}\n")
    else:
        print(f"non_local = {prm.non_local}, local Sobolev source assumed, with no interactions between lines")

    return delta_v_mat, int_vmax, self_cp, inter_cp


# ----------------------------------------------
def define_frequency_grid(i_vmax, line_d_width):
    # ------------------------------------------
    """
    define the frequency/velocity grid
    we use a common velocity reference frame for all computed lines
    which means that we must define a wider frequency grid to allow
    for the different wavelengths of the lines
    """

    # define the frequency grid arrays
    dfreql = np.zeros(prm.nline)
    profilewidth = np.zeros(prm.nline)
    bxmaxl = np.zeros(prm.nline)
    nwingl = np.zeros(prm.nline, int)

    for lx in range(prm.nline):
        # number of velocity grid points per side of local absortion line profile
        nwingl[lx] = int(line_d_width[lx] / prm.ibeta)
        # elementary frequency step (in units of line Doppler widths)
        dfreql[lx] = 2 * prm.linewing / nwingl[lx]
        # profile graph abscissa unit
        profilewidth[lx] = grid.vmax + prm.linewing * line_d_width[lx]
        bxmaxl[lx] = profilewidth[lx] / grid.vmax
        if prm.debug_mode:
            print(f"Line {lx} - line_d_width[{lx}] = {line_d_width[lx]:.2f} km/s")
            print(f"nwingl[{lx}] = {nwingl[lx]}, dfreql[{lx}] = {dfreql[lx]}")
            print(f"profilewidth[{lx}] = {profilewidth[lx]} km/s")

    # -----------------------------------------------------
    # now define a common reference frame in velocity units
    # -----------------------------------------------------
    # maximum number of points per local line wing
    nwing = int(np.amax(nwingl) * prm.linewing)
    # elementary frequency grid interval (in units of line Doppler widths)
    dfreq = 2 * prm.linewing / nwing
    # elementary wavelength grid interval (in angstroms)
    dwave = const.cvel / (dfreq / const.km_s) * const.angstrom
    # print(const.cvel, dfreq, const.km_s, const.angstrom, dwave)

    # profile graph abscissa unit
    bxmax = np.around(np.amax(bxmaxl), 2)
    # maximum local absorption profile width
    awidth = int(np.amax(line_d_width))
    # elementary velocity step
    dvel = awidth / nwing  # in km/s
    # number of frequencies per line side including maximum line separation
    nfreq = int((np.amax(profilewidth) + i_vmax) / dvel)
    # total number of frequencies
    nfreq_tot = 2 * nfreq + 1
    print(f"Maximum displacement between interacting lines i_vmax = {i_vmax} km/s, nfreq = {nfreq}, "
          f"nfreq_tot = {nfreq_tot}")

    # -----------------------------------------------------------------------
    # next, we compute the local absorption profile integrals for the lines
    # Because line absorption profiles may have different widths,
    # we use the individual local profile integrals instead
    # of a common one as was done in the case of independent lines
    # -----------------------------------------------------------------------
    # For the Sobolev approximation, set nwing to 0 at this point in the code
    # -----------------------------------------------------------------------

    if prm.sobolev_mode:
        nwing = 0
        for lx in range(prm.nline):
            # nwingl[lx, :] = 0
            nwingl[lx] = 0
    nwing2 = 2 * nwing if nwing else 1

    print(f"awidth = {awidth:.2f} km/s, nwing = {nwing}, dvel = {dvel} km/s, dfreq = {dfreq:.2f}, bxmax = {bxmax}\n")

    dxd = np.zeros((prm.nline, nwing2))
    # line_flux = np.zeros((prm.nline, nfreq_tot))

    # ------------------------------------------
    # compute local absorption profile integrals
    # ------------------------------------------

    for lx in range(prm.nline):
        # noinspection PyTypeChecker
        nwingl[lx] = min(nwingl[lx], nwing)
        dxd[lx, :] = variable_local_profile_integral(nwing, nwingl[lx], dfreql[lx]) if nwing else 1.0
        if prm.debug_mode:
            print(f"dxd[{lx},:] \n {dxd[lx, :]}")
            print(f"Integral of local profile function (sum(dxd[{lx}])) = {np.sum(dxd[lx]):.4f}")

    if prm.graph_mode:
        x_axis = [j-nwing for j in range(2 * nwing)]
        plres.plot_absorption_profiles(x_axis, dxd)

    # --------------------------------------------------------------------------------------------------
    # finally, define velocity dictionary {index: velocity} for all velocities found in the line profile
    # --------------------------------------------------------------------------------------------------

    f_dict = {k: dvel * k for k in range(-nfreq, nfreq + 1)}  # if abs(dfreq*k) <= vmax}
    # print(f"f_dict_unsorted \n {f_dict_unsorted} \n") if prm.flux_debug_mode else None
    # f_dict = collections.OrderedDict(sorted(f_dict_unsorted.items()))
    print(f"f_dict \n {f_dict} \n") if prm.flux_debug_mode else None

    return nfreq, nfreq_tot, nwing, dwave, dvel, bxmax, dxd, f_dict
