"""
module name: in_out_library. Functions in this module are for storing or reading computation results
"""

import numpy as np
import import_parameters as prm
import set_grids as grid
import function_library as lib
import constants as const


# ------------------------------
def read_source(in_file_source):
    # --------------------------
    """
    reads the line source functions, normalized to the local continuum flux, as a function of radius
    """
    radius = np.zeros(prm.idr)
    source_l = np.zeros((prm.nline, prm.idr))
    source_nl = np.zeros((prm.nline, prm.idr))
    # color = ['b', 'r', 'g', 'c', 'm']
    with open(in_file_source, "r") as f:
        while True:
            one_line = f.readline()
            if not one_line:
                break
            v0 = one_line  # one_line.split()
            lx = int(v0)
            # print(f"\nlx = {lx}\n")
            for i in range(prm.idr):
                rvalue, xvalue, yvalue = f.readline().split()
                # print(f"r = {rvalue}, local_source = {xvalue}, source = {yvalue}")
                radius[i] = float(rvalue)
                source_l[lx, i] = float(xvalue)
                source_nl[lx, i] = float(yvalue)
        return radius, source_l, source_nl


# -------------------------------------
def read_flux(in_file_flux, nfreq_tot):
    # ---------------------------------
    """
    reads the line profile flux, normalized to the local continuum flux, as a function of wavelength
    individual line profiles and their sum are shown on the same scale.
    """
    x_axis = np.zeros((prm.nline, nfreq_tot))
    flux = np.zeros((prm.nline, nfreq_tot))
    fc = np.zeros(prm.nline)
    with open(in_file_flux, "r") as f:
        while True:
            one_line = f.readline()
            if not one_line:
                break
            x0, x1 = one_line.split()
            lx = int(x0)
            fc[lx] = float(x1)
            # print(f"\nlx = {lx}, fc[{lx}] = {fc[lx]}\n")
            for k in range(nfreq_tot):
                v0, v1 = f.readline().split()
                x_axis[lx, k] = float(v0)
                flux[lx, k] = float(v1)
                # print(f"wavelength = {v0}, flux = {v1}")

    return x_axis, flux, fc


# ------------------------------------------------------------
def save_parameters(outfile_prm, date, nti, tempi, exec_time):
    # --------------------------------------------------------
    """
    store computation parameters
    """
    with open(outfile_prm, "w") as f:
        f.write(f"Time stamp = {date}, nt = {nti:.2e}, temp = {tempi:.2f}\n")
        f.write('\nSpatial grid parameters\n')
        f.write(f"idr = {prm.idr}, log_rgrid = {prm.log_rgrid}, idz = {prm.idz} "
                f"\nidc = {prm.idc}, ide = {prm.ide}")
        f.write('\nStar, envelope, and velocity field parameters\n')
        f.write(f"rc = {prm.r_star / const.R_sun} R_sun, mc = {prm.mc} M_sun, teff = {prm.teff}, "
                f"rmax = {prm.rmax} rc\n")
        f.write(f"velocity_index = {prm.velocity_index}, alpha = {prm.alpha}, vc = {grid.v_env[0]:.2e} km/s "
                f"vturb_env = {prm.vturb0 * const.km_s:.0f} km/s")
        if prm.one_model:
            f.write(f"\nntc = {prm.ntc:.2e} cm^-3, temp_env = {prm.temp0:.2e} K")
        else:
            f.write(f"\nProduction mode\nint0 = {prm.int0}, itemp0 = {prm.itemp0}, i_nt_min = {prm.i_nt_min}, "
                    f"i_nt_max = {prm.i_nt_max}, i_temp_min = {prm.i_temp_min}, i_temp_max = {prm.i_temp_max}\n"
                    f"nt domain = [{prm.nt_min:.2e}, {prm.nt_max:.2e}], "
                    f"temp domain = [{prm.temp_min}, {prm.temp_max}]\n")
        f.write(f"\nList of elements and lines considered\n")
        for index, element in enumerate(prm.elements):
            f.write(f"{element[0]}, line_range = {prm.line_range[index]}\n")
        f.write(f"Total number of computed lines = {prm.nline}")
        f.write('\nSource function parameters\n')
        f.write(f"interact_mode = {prm.interact_mode}, non_local = {prm.non_local}, epsilon_flag = {prm.epsilon_flag} ")
        f.write(f"igauss_core = {prm.igauss_core}, igauss_shell = {prm.igauss_shell}")
        f.write('\nFrequency grid parameters\n')
        f.write(f"Sobolev_mode = {prm.sobolev_mode}, velocity resolution ibeta = {prm.ibeta} km/s, "
                f"line wing width = {prm.linewing} x max velocity\n")

        f.write(f"\nApple M1 Pro 1 core CPU time = {exec_time:.2f} seconds")
    print(f"computation parameters stored in {outfile_prm}")


# -----------------------------------------------
def save_sources(outfile_s, source_l, source_nl):
    # -------------------------------------------
    """
    store local and non_local sources as a function of radius
    """

    with open(outfile_s, "w") as f:
        for lx in range(prm.nline):
            out = str(lx) + '\n'
            f.write(out)
            for i in range(prm.idr):
                out = '{0}  {1}  {2}\n'.format(str(grid.r[i]), str(source_l[lx, i]), str(source_nl[lx, i]))
                f.write(out)
    print(f"line sources stored in {outfile_s}")


# -------------------------------------------------------------------------------
def save_profiles(outfile_p, outfile_b, lambda0, nfreq, dvel, emergent_flux, fc):
    # ---------------------------------------------------------------------------
    """
    store line profiles as a function of wavelength
    store iron profile and blend separately
    """
    nfreq_tot = 2 * nfreq + 1
    x_axis = np.zeros((prm.nline, nfreq_tot))
    for lx in range(prm.nline):
        x_axis[lx] = [lambda0[lx] * (1.0 + j * dvel / (const.km_s * const.cvel)) / const.angstrom
                      for j in range(-nfreq, nfreq + 1)]
    with open(outfile_p, "w") as f:
        for lx in range(prm.nline):
            f.write(str(lx) + '  ' + str(fc[lx]) + '\n')
            for wav in range(nfreq_tot):
                out = '{0}  {1}  \n'.format(str(x_axis[lx, wav]), str(emergent_flux[lx, wav]))
                f.write(out)
    print(f"Line profiles stored in {outfile_p}")
    region = 'all'
    x_axis_blend, dwave, resampled_flux, blend, blend_fc = lib.resample_flux(region, lambda0, nfreq,
                                                                             nfreq_tot, dvel, emergent_flux, fc)
    lx = 1
    nfreq_blend = len(x_axis_blend)
    with open(outfile_b, "w") as f:
        f.write(str(lx) + '  ' + str(nfreq_blend) + '\n')
        for wav in range(nfreq_blend):
            out = '{0}  {1}  \n'.format(str(x_axis_blend[wav]), str(resampled_flux[lx, wav]))
            f.write(out)
        f.write(str(prm.nline) + '  ' + str(nfreq_blend) + '\n')
        for wav in range(nfreq_blend):
            out = '{0}  {1}  \n'.format(str(x_axis_blend[wav]), str(blend[wav]))
            f.write(out)
    print(f"Iron profile and blend stored in {outfile_b}")


# -------------------------------------------------------
def save_cp_surfaces(outfile_cp, rplots, igauss, x0, y0):
    # ---------------------------------------------------
    """
    store approximate cp and ri surfaces in cartesian coordinates
    """
    with open(outfile_cp, "w") as f:
        for rplot, color in rplots.items():
            for lx in range(prm.nline):
                for lxx in range(prm.nline):
                    out = str(rplot) + ' ' + color + ' ' + str(igauss) + ' ' + str(round(grid.r[rplot], 2)) + ' ' + str(
                        0.0) \
                          + ' ' + str(lx) + ' ' + str(lxx) + '\n'
                    f.write(out)
                    for j in reversed(range(igauss)):
                        out = str(round(x0[lx, lxx, rplot, j], 2)) + ' ' + str(round(y0[lx, lxx, rplot, j], 2)) + '\n'
                        f.write(out)

    print(f"cp_surfaces stored in {outfile_cp}")


# -----------------------------------------------------------------------------------------------------
def store_equivalent_widths(out_file_ew, wavel, nfreq, nfreq_tot, dvel, nt, temp, flux, fc):
    # -------------------------------------------------------------------------------------------------
    """
    This version is limited to the 3-line case
    """
    out = []
    region = 'all'
    x_axis_blend, dwave, resampled_flux, blend, average_fc = lib.resample_flux(region, wavel, nfreq,
                                                                               nfreq_tot, dvel, flux, fc)
    ew = lib.compute_equivalent_widths(resampled_flux, fc, blend, average_fc, dwave)

    with open(out_file_ew, "a") as f:
        out.append(nt)
        out.append(temp)
        # recall that ew[-1] is the lambda3969 blend EW
        for lx in range(prm.nline + 1):
            out.append(ew[lx])
        f.write(str(out)+'\n')


