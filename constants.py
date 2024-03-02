"""
module name: constants
gives some useful physical and astronomical constants in cgs units
"""
import scipy.constants as cst

angstrom = 1.0e-8  # 1 angstrom in cm
km_s = 1.0e-5  # 1cm/s in km/s
eV = 1.6021766208e-12  # erg
h = 6.62607015e-27  # Planck constant en erg*s
cvel = 29979245800.0  # velocity of light in vacuum en cm/s
k_B = 1.380649e-16  # Boltzmann constant en erg/K
k = k_B / eV  # Boltzmann constant in eV
sigma = 5.6703744191844314e-05  # Stefan-Boltzmann constant en g / (K4 s3)
G = 6.674299999999999e-08  # gravitational constant en cm3 /(g*s2)
m_e = 9.1093837015e-28  # electron mass
m_p = 1.67262192369e-24  # proton mass
m_n = 1.67492749804e-24  # neutron mass
e = 4.80320425e-10  # electron charge in cgs units
# e = 1.602176634e-19 # electron charge in Coulomb
L_sun = 3.828e+33  # solar luminosity en erg/s
M_sun = 1.988409870698051e+33  # solar mass en g
R_sun = 69570000000.0  # solar radius en cm
au = 14959787070000.0  # astronomical unit en cm
pc = 3.085677581467192e+18  # parsec en cm
pi = cst.pi  # good ole pi
saha_cst = 2.0 * pi * m_e * k_B / (h * h)  # cgs units
tau_cst = pi * e * e / (m_e * cvel)  # = 0.026540083433884684 (Gauss's system)
inv_cm_to_ev = 1.239841984332e-4  # cm-1 to eV conversion factor
# print(f"tau_cst = {tau_cst}")
year = 365.25*24*3600

ly_alpha = 1215.67
ha = h*cvel/(ly_alpha*angstrom*eV)
balmer = [3797.90,3835.38,3889.05,3970.07,4101.73,4340.46,4861.32,6562.80]
for b in balmer:
    hb = h*cvel/(b*angstrom*eV)+ha
    #print(ha, hb)
