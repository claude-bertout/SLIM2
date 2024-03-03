"""
Module name: input_parameters
This modules reads the parse configuration parameters from text file computation_parameters.conf
and derives many of the secondary computation parameters from these. The main computation
is further prepared by modules define_grids and line_data.
"""
import numpy as np
import csv
from sys import platform
# from pathlib import Path

import parameter_parser as parse
import constants as const
from constants import G, M_sun, R_sun

# all input parameters are defined in the computation_parameters.conf file
# location of files (to be adapted according to computing platform format)
file_location = parse.config['project_directory']
file_location_base = file_location + "results/"

# computational option parameters
debug_mode = parse.config['debug_mode']
silent_mode = parse.config['silent_mode']
lte_debug_mode = parse.config['lte_debug_mode']
source_debug_mode = parse.config['source_debug_mode']
flux_debug_mode = parse.config['flux_debug_mode']
graph_mode = parse.config['graph_mode']
#
# functional input parameters
one_model = parse.config['one_model']
interact_mode = parse.config['interact_mode']
non_local = parse.config['non_local']
epsilon_flag = parse.config['epsilon_flag']
frac = parse.config['frac']
frac2 = frac * frac
eps = parse.config['eps']
conv = parse.config['conv']
iconv_max = parse.config['iconv_max']
core_only = parse.config['core_only']
env_only = parse.config['env_only']

# production run parameters
if not one_model:
    print(f"\none_model = {one_model}")
    # number of nt's
    int0 = parse.config['int0']
    # number of temp's
    itemp0 = parse.config['itemp0']
    # nt domain
    nt_min = parse.config['nt_min']
    nt_max = parse.config['nt_max']
    # temp domain
    temp_min = parse.config['temp_min']
    temp_max = parse.config['temp_max']
    # divide computation in 4 series of 100 models each, with subsets [0:10] and [10:20] for each variable
    # define nt range
    i_nt_min = parse.config['i_nt_min']
    i_nt_max = parse.config['i_nt_max']
    # define temp range
    i_temp_min = parse.config['i_temp_min']
    i_temp_max = parse.config['i_temp_max']
    print(f"Entering production mode\nint0 = {int0}, itemp0 = {itemp0}, i_nt_min = {i_nt_min}, i_nt_max = {i_nt_max}, "
          f"i_temp_min = {i_temp_min}, i_temp_max = {i_temp_max}\nnt domain = [{nt_min:.2e}, {nt_max:.2e}], "
          f"temp domain = [{temp_min}, {temp_max}]")

# stellar input parameters
rc = parse.config['rc']  # rc is not a true parameter, as it is always normalized to 1 in the computation
mc = parse.config['mc']
teff = parse.config['teff']
#
# envelope input parameters
rmax = parse.config['rmax'] * rc
r_star = parse.config['r_star'] * R_sun
velocity_index = parse.config['velocity_index']
alpha = parse.config['alpha']

vc = np.sqrt(2.0 * G * mc * M_sun / r_star) * const.km_s  # escape velocity at the stellar surface

ntc = parse.config['ntc']
temp0 = parse.config['temp0']
vturb0 = parse.config['vturb0']

print('\nStar, envelope and velocity field parameters')
print(f"R_star = {r_star/R_sun} R_sun, mc = {mc} M_sun, teff = {teff:.2f} K,"
      f"Rc = {rc}, Rmax = {rmax}Rc\n"
      f"velocity_index = {velocity_index}, alpha = {alpha}, v_esc = {vc:.2f} km/s, ntc = {ntc:.2e} cm^-3, "
      f"temp_env = {temp0:.2f} K, vturb0 = {vturb0 * 1.0e-5:.2f} km/s\n")

# source function parameters
print('Source functional parameters')
print(f"interact_mode = {interact_mode}, non_local = {non_local}, epsilon_flag = {epsilon_flag}, frac = {frac}\n")

#
# dimensional parameters 
log_rgrid = parse.config['log_rgrid']  # choose log or linear radial grid
idr = parse.config['idr']  # r-grid
idz = parse.config['idz']  # z-grid from -zmax to zmax
idc = parse.config['idc']  # p-grid from 0 ro rc
ide = parse.config['ide']  # p-grid from rc ro rmax
igauss_core = parse.config['igauss_core']  # orders of gaussian integration
igauss_shell = parse.config['igauss_shell']
#
print('Grid parameters')
print(f"log_rgrid = {log_rgrid}, idr = {idr}, log_rgrid = {log_rgrid}, idz = {idz}, idc = {idc}, ide = {ide} \n")
#
print('Gaussian integrations parameters for source')
print(f"igauss_core = {igauss_core}, igauss_shell = {igauss_shell} \n")

# import line list and define various constants for the line_data and lte modules

# input atoms and lines under consideration
element_list = parse.config['elements']
line_list = parse.config['nlines']
print(f"element_list = {element_list}")
print(f"line_list = {line_list}")

eparser = csv.reader(element_list)
lparser = csv.reader(line_list)
elem = []
elin = []
for fields in eparser:
    for element_list in fields:
        elem.append(element_list)
# print(f"elem = {elem}")
for fields in lparser:
    for line_list in fields:
        elin.append(int(line_list))
# print(f"elin = {elin}")
elements = []
for i in range(len(elem)):
    elements.append((elem[i], elin[i]))
# print(elements)

nline = sum(elin)
nelem = len(elements)
start = 0
stop = 0

line_range = np.zeros(nelem, object)
atomic_weight = np.zeros(nelem)

for index, element in enumerate(elements):
    # print(f"index = {index}, element = {element}")
    if index == 0:
        start = index
        # warning here is a known PyCharm bug still relevant on 2023.2.3
        stop = element[1]
    else:
        # noinspection PyTypeChecker
        start = start + elements[index - 1][1]
        # noinspection PyTypeChecker
        stop = stop + elements[index][1]
    line_range[index] = np.arange(start, stop)
    # print(f"start = {start}, stop = {stop}")

print('Elements under consideration')
for index, element in enumerate(elements):
    # warning here is a known PyCharm bug still relevant on 2023.2.3
    print(f"{element[0]}, line_range[{index}] = {line_range[index]}")
print(f"Total number of computed lines = {nline}\n")

# frequency grid parameters
sobolev_mode = parse.config['sobolev_mode']
ibeta = parse.config['ibeta']  # number of points per Doppler width (i.e., computational resolution in km/s)
linewing = parse.config['linewing']  # width of line wing in units of velocity widening

print(f"Frequency grid parameters\nsobolev_mode = {sobolev_mode}, velocity resolution ibeta = {ibeta} km/s, "
      f"line wing width = {linewing} x max velocity\n")
