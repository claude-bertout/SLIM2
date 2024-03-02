"""
Module name: input_grid_data
This modules sets up the spatial grids r, p_core, p_env used in source and flux computations
as well as the runs of radial physical quantities (velocity v_env, temp, proton number density ns)
The envelope velocity gradient v_grad and the nature of the flow are also analyzed here
"""

import numpy as np
import import_parameters as prm
from scipy import interpolate

# import matplotlib.pyplot as plt
global rd, v_env_rd, r_mid, r_s, i_mid, i_s


# -------------------------------
def func_interp_spline(rx, x, y):
    # ---------------------------
    # f = interpolate.CubicSpline(x, y)
    rx = np.where(rx < prm.rc, prm.rc, rx)
    rx = np.where(rx > prm.rmax, prm.rmax, rx)
    f = interpolate.interp1d(x, y, kind='linear')
    return f(rx)


# --------------------------------------------------
def define_velocity(vel_index: int, rx: np.ndarray):
    # ----------------------------------------------
    global r_mid, r_s, i_mid, i_s
    """
    v_law is the parameterized envelope velocity law
    """
    # data of HEA Model 5
    # r_wind = np.array([1.0, 1.03, 1.07, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0, 5.0, 7.5, 10.001])
    # v_wind = np.array([2.0, 3.2, 7.2, 13.0, 26.0, 46.0, 58.0, 81.0, 129.0, 160.0, 188.0, 213.0, 214.0, 215.0])
    # velocity: np.ndarray = prm.vc * (1.0 - 0.98 * prm.rc / rx)  # Olson
    # velocity: np.ndarray = prm.vc * (1.0 - 0.999 * prm.rc / rx) ** 0.5  # modified Castor
    #
    velocity = np.zeros(prm.idr)
    # approx. power law fit of HEA Model 5
    if vel_index == 0:
        # accelerating outflow
        param = np.array([250.74915422, 0.9816051, 0.95327265])
        velocity = param[0] * (1.0 - param[1] / rx) ** param[2]
    elif vel_index == 1:
        # accretion flow
        velocity: np.ndarray = prm.vc * (prm.rc / rx) ** prm.alpha
    elif vel_index == 2:
        # non-monotonic outflow
        i_mid = int(prm.idr / 5)
        r_mid = r[i_mid]

        param = np.array([400, 0.9816051, 0.95327265])
        v_max = param[0] * (1.0 - param[1] / r_mid) ** param[2]
        velocity = np.where(rx <= r_mid, param[0] * (1.0 - param[1] / rx) ** param[2],
                            v_max * (r_mid / rx) ** prm.alpha)
    elif vel_index == 3:
        # non-monotonic accretion flow
        # print(f"rx \n{rx}")
        i_s = int(prm.idr / 20)
        r_s = rx[i_s]
        r_ss = rx[i_s - 1]
        # print(f"r_ss = {r_ss}")
        # print(f"Non_monotonic accretion flow \nr_s = {r_s:.2f} rc")
        v_max = prm.vc * (prm.rc / r_s) ** prm.alpha
        # print(f"v_max = {v_max}")
        v_ss = v_max / 4.0
        # print(f"v_ss = {v_ss}")
        velocity[i_s - 1] = v_ss
        d = (v_ss - 1.0e1) / (r_ss - rx[0])
        # print(f"d = {d}")
        velocity = np.where(rx <= r_ss, 1.0e1 + d * (rx - rx[0]), v_max * (r_s / rx) ** prm.alpha)
        # print(velocity)

    elif vel_index == 4:
        # decelerating ballistic outflow
        velocity: np.ndarray = prm.vc * (prm.rc / rx) ** prm.alpha

    else:
        print('Velocity field not specified')
        exit()

    return velocity


# -----------
def v_law(rx):
    # -------
    # smoothed velocity law to avoid numerical noise in source functions
    # rd and v_env_rd are global variable in this module
    return func_interp_spline(rx, rd, v_env_rd)


# --------------------------------------------------------------------------------
# all input parameters are defined in the transfer.conf file and imported from prm
# --------------------------------------------------------------------------------

# r is the vector of radial points defined from rc to rmax
drc = np.log(prm.rc)
drmax = np.log(prm.rmax)
if prm.log_rgrid:
    r = np.exp(np.linspace(drc, drmax, prm.idr))  # logarithmic r-grid
else:
    r = np.linspace(prm.rc, prm.rmax, prm.idr)  # linear r-grid

r[0] = r[0] * (1.0 + prm.frac)
# r[-1] = r[-1] * (1.0 - prm.frac)
print(f"r-grid \n{r}") if prm.debug_mode else None

# plot results for 5 or 3 radial points depending on radial point number.
# The dictionary  rplots indicates both plot color and associated radial point.
if len(r) >= 100:
    rplots = {5: 'b', int(len(r) / 4): 'r', int(len(r) / 2): 'k', int(3 * len(r) / 4): 'm', len(r) - 4: 'g'}
else:
    rplots = {1: 'b', int(len(r) / 2): 'r', int(len(r) - 1): 'g'}

print(f"cp_surfaces will be shown for r = {[np.round(r[i], 2) for i in rplots.keys()]}\n")

# --------------------------------------------------
# temperature and turbulent velocity in the envelope
# --------------------------------------------------
temp = np.ones(prm.idr) * prm.temp0
vturb = np.ones(prm.idr) * prm.vturb0

# v_env_rd is the envelope velocity on a subset rd of the radial grid r

# added 26/01/23 first for non-monotonic case, then extended to all velocities:
# we need to avoid discontinuities in the velocity and its derivatives, since they
# introduce spurious discontinuities in the source function, so we define the velocity
# grid v_env, which is the cubic-spline-interpolated envelope velocity vector
# on the radial grid.
if prm.log_rgrid:
    rd = np.exp(np.linspace(drc, drmax, int(prm.idr / 2)))  # logarithmic r-grid
else:
    rd = np.linspace(prm.rc, prm.rmax, int(prm.idr / 2))
v_env_rd = define_velocity(prm.velocity_index, rd)
v_env = v_law(r)

print(f"Flow properties:")
print(f"v-grid in km/s \n{v_env}") if prm.debug_mode else None

if prm.velocity_index == 0:
    print('Outflowing, accelerating gas')
elif prm.velocity_index == 1:
    print('Infalling gas')
elif prm.velocity_index == 2:
    print(f"Non-monotonic outflow with r_mid = {r_mid:.2f} Rc at i_mid = {i_mid}")
elif prm.velocity_index == 3:
    print(f"Accretion shock with R_s = {r_s:.2f} Rc at i_s = {i_s}")
elif prm.velocity_index == 4:
    print('Outflowing, decelerating gas')
else:
    print('Undefined velocity field. Abort.')
    exit()

vmax = np.round(np.max(np.abs(v_env)), 2)
v0 = np.round(np.abs(v_env[0]), 2)
v1 = np.round(np.abs(v_env[-1]), 2)
print(f"abs(v_env[0]) = {v0} km/s, abs(v_env[-1]) = {v1} km/s, abs(vmax) = {vmax}")
if vmax == v1:
    accel_flag = 'True'
elif vmax == v0:
    accel_flag = 'False'
else:
    accel_flag = 'Both'
print(f"accel_flag = {accel_flag}")

if accel_flag == 'True':
    print("Flow is accelerating outwards")
elif accel_flag == 'False':
    print("Flow is accelerating inwards")
else:
    print('Flow is non-monotonic')
# noinspection PyTypeChecker
v_grad = max(v_env) - min(v_env[-1], v_env[0])
print(f"Envelope maximum velocity gradient v_grad = {v_grad:.2f} km/s\n")

# ---------------------------------------------------------------------------
# nt is the total number of particles (atoms+ions+electrons) vector on r-grid
# we assume mass conservation in the flow (mass of electrons is neglected)
# ---------------------------------------------------------------------------

# nt_rd = prm.ntc * v_env_rd[0] * (prm.rc / rd) ** 2.0 / v_env_rd
# nt = func_interp_spline(r, rd, nt_rd)
nt = prm.ntc * np.abs(v_env[0]) * (prm.rc / r) ** 2.0 / np.abs(v_env)
print(f"nt-grid \n{nt}") if prm.debug_mode else None
