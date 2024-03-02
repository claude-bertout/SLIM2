"""
library of general purpose functions for source and line profile computations
"""

import numpy as np
import import_parameters as prm
import constants as const
from constants import cvel
import set_grids as grid

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


# -------------------------
# radial velocity functions
# -------------------------
# -----------------
def vrad_z(rx, zx):
    # -------------
    """
    vrad_z returns the radial velocity in direction z for all z's at p=cst
    valid for any velocity field
    """
    vx = np.interp(rx, grid.r, grid.v_env)
    cos_a = zx / rx
    return vx * cos_a


# -------------------------
def der_z_num(rp, z):
    # ---------------------
    """
    calculates numerically the derivative of vrad at rp in direction z
    valid for any velocity field
    """
    z = np.where(z != 0, z, prm.eps)
    vel1 = vrad_z(rp, z)
    vel2 = vrad_z(rp, z * (1.0 + prm.frac))
    return np.abs((vel2 - vel1) / (prm.frac * z))


# -------------------------------------
def vrads(r, sth, cth, stheta, ctheta):
    # ---------------------------------
    """same as vrads_1; radial velocity at (r, th) in direction theta"""
    costhetap = ctheta * cth + stheta * sth
    vrad = grid.v_law(r) * costhetap
    return vrad


# -------------------------------------
def vrads_0(params):
    # ---------------------------------
    """same as vrads_1; radial velocity at (r, th) in direction theta"""
    (r, sth, cth, stheta, ctheta) = params
    costhetap = ctheta * cth + stheta * sth
    vrad = grid.v_law(r) * costhetap
    return vrad


# ------------------------------
def Q_l(igauss, stheta, ctheta):
    # --------------------------
    """
    returns the absolute value of radial velocity derivative Q = dv_l/dl at r = rad
    in direction lx defined by theta. Spherical symmetry assumed.
    (see eq. 14 of Hummer and Rybycki 1978)
    called by compute_eta_local_term in source module
    """

    # compute radial velocities in all directions at r0
    rpz = np.zeros((prm.idr, igauss))
    for i in range(prm.idr):
        rpz[i, :] = grid.r[i]
    print(f"from Q - local - rpz \n{rpz}") if prm.source_debug_mode else None
    dr = prm.frac * rpz
    zr = rpz
    pr = 0.0

    # compute radial velocities in all direction at rad + dr
    zrp = rpz + ctheta * dr
    prp = stheta * dr
    deriv = derq(zr, zrp, pr, prp, dr, ctheta, stheta)

    if prm.source_debug_mode:
        print(f"1/deriv \n{1.0 / deriv}")
        print(f"shape(deriv) = {np.shape(deriv)}")

    return deriv


# ------------------------------------------------
def derq(z1, z2, p1, p2, dr, stheta, ctheta):
    # --------------------------------------------
    """
    returns the velocity derivative in theta direction.
    Called by escape functions
    """

    r1 = np.sqrt(z1 * z1 + p1 * p1)
    ct1 = z1 / r1
    st1 = p1 / r1

    r2 = np.sqrt(z2 * z2 + p2 * p2)
    ct2 = z2 / r2
    st2 = p2 / r2

    v1 = grid.v_law(r1)
    v2 = grid.v_law(r2)

    pr1 = st1 * stheta + ct1 * ctheta
    pr2 = st2 * stheta + ct2 * ctheta

    vr1 = v1 * pr1
    vr2 = v2 * pr2

    dera = (vr2 - vr1) / dr

    return np.abs(dera)


# ------------------------------
def Planck_function(wave, temp):
    # --------------------------
    """
    returns the Planck function in cgs units. 09/2019 version
    """
    freq = cvel / wave
    aa = 1.4744994402842402e-47  # 2*h/c**2
    cc = 4.7992446622113495e-11  # h/k
    dx = np.divide(cc * freq, temp)
    return aa * freq ** 3 / (np.exp(dx) - 1)


# ------------------------------------------
def ctheta_grid(reverse_flag, igauss, a, b):
    # --------------------------------------
    # sets up the cos(theta) grid for the Gaussian integration over space
    # r is the radial grid vector

    ctheta = np.zeros((prm.idr, igauss))

    # compute abscissa and weights of the Gaussian integration
    # a and b are the lower and upper integration limits
    # igauss is the order of the Gaussian approximation
    xgauss, wgauss = np.polynomial.legendre.leggauss(igauss)
    if reverse_flag:
        xgauss = xgauss[::-1]
        wgauss = wgauss[::-1]
    for i in range(prm.idr):
        ctheta[i, :] = 0.5 * (b[i] - a[i]) * xgauss[:] + 0.5 * (a[i] + b[i])
        # print(f"from ctheta_grid - theta[{i}, :] \n{np.rad2deg(np.arccos(ctheta[i, :]))}")

    return ctheta, wgauss


# ------------------------
def find_sign_change(arr):
    # --------------------
    result = []
    vp = arr[0]
    for i, v in enumerate(arr):
        if i == 0:
            change = False
        elif v < 0 < vp:
            change = True
        elif v > 0 > vp:
            change = True
        else:
            change = False
        result.append(change)
        vp = arr[i]
    return np.array(result)


# ---------------------
def smooth(y, box_pts):
    # -----------------
    """
    :param y: data vector to be smoothed
    :param box_pts: smoothing window
    :return: convolved array
    NB. the smoothed array is resized to the size of the input array;
    this eliminates boundary effects
    """
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='full')
    ylen = len(y)
    yslen = len(y_smooth)
    ysmask = np.ones(len(y_smooth), dtype=bool)
    ysmask[[i for i in range(ylen, yslen)]] = False
    y_smooth = y_smooth[ysmask]

    return y_smooth


# -------------------------------------------------------------------
def resample_flux(region, wavel, nfreq, nfreq_tot, dvel, flux, fc):
    # ---------------------------------------------------------------
    """
    resamples flux on a common wavelength scale and computes the blend
    """
    x_axis_flux = np.zeros((prm.nline, nfreq_tot))
    resampled_flux = np.zeros((prm.nline, nfreq))
    blend = np.zeros(nfreq)

    for lx in range(prm.nline):
        x_axis_flux[lx, :] = [wavel[lx] * (1.0 + j * dvel / (const.km_s * const.cvel)) / const.angstrom
                              for j in range(-nfreq, nfreq + 1)]

    x_min = np.min(x_axis_flux[:, 0]) * (1.0 - prm.frac)
    x_max = np.max(x_axis_flux[:, -1]) * (1.0 + prm.frac)
    x_axis_blend = np.linspace(x_min, x_max, nfreq)
    average_fc = np.average(fc[:])
    dwave = abs(x_max - x_min) / (nfreq - 1)

    for lx in range(prm.nline):
        # smooth flux a little to avoid high frequency numerical noise in accretion flow case
        # if prm.velocity_index == 1 or prm.velocity_index == 3:
        # flux[lx, :] = savgol_filter(flux[lx, :], 21, 3)  # window size 5, polynomial order 3
        resampled_flux[lx, :] = np.interp(x_axis_blend, x_axis_flux[lx, :], flux[lx, :])
        resampled_flux[lx, :] = smooth(resampled_flux[lx], 11)  # box convolution with window size 11 data points

    if region == 'envelope':
        for k in range(nfreq):
            blend[k] = np.sum(resampled_flux[:, k])
    else:
        for k in range(nfreq):
            blend[k] = np.sum(resampled_flux[:, k] - average_fc) + average_fc
    blend = np.where(blend < 0, 0, blend)

    return x_axis_blend, dwave, resampled_flux, blend, average_fc


# -------------------------------------------------------------------------
def compute_equivalent_widths(emergent_flux, fc, blend, average_fc, dwave):
    # ---------------------------------------------------------------------
    """
    By (historical) convention, equivalent widths are positive for absorption, negative for emission.
    The quantity computed here is the net equivalent width integrated over the entire
    emission + absorption profile.
    """
    integ_flux = np.zeros(prm.nline)
    ew = np.zeros(prm.nline + 1)
    for lx in range(prm.nline):
        integ_flux[lx] = np.trapz(y=fc[lx] - emergent_flux[lx], x=None, dx=dwave)
        ew[lx] = integ_flux[lx] / fc[lx]
        print(f"Equivalent width for lx = {lx} - EW = {np.round(ew[lx], 2)} Angstrom")

    integ_blend = np.trapz(y=average_fc - blend, x=None, dx=dwave)
    ew[-1] = integ_blend / average_fc
    print(f"Equivalent width for the blend - EW = {np.round(ew[-1], 2)} Angstrom")

    return ew
