"""
Module name : LTE_module
This module computes the LTE electron density and level populations for
known total number density and temperature distribution.
Solar composition is assumed.
Electron donor metals taken into account: C, Si, Fe, Mg, Ni, Cr, Ca, Na, K.
Caution: only one ionization level of the above elements (plus H and He)
is taken into account.
The derivation follows Gray (Stellar photospheres, p. 181 ff.)
"""

import numpy as np
# import matplotlib.pyplot as plt
import csv
from constants import k, saha_cst, tau_cst
import import_parameters as prm
import import_atomic_data as adat

# global atom, ion_level, a_weight, abund, lambda0, el_ev, eu_ev, gl, gu, flu, Aul, log_gf, ion_level_l, ion_level_u


# ----------------------------------------
def boltzmann_ratio(gb, ga, eb, ea, temp):
    # ------------------------------------
    """
    solves the Boltzman equation.
    """
    exponent = -(eb - ea) / (k * temp)
    ratio = (gb / ga) * np.exp(exponent)
    # print(f"Boltzmann exponent = {exponent}, ea = {ea}, ga = {ga}, eb = {eb}, gb = {gb}, ratio = {ratio}")
    return ratio


# ----------------------------
def polyn_approx(g0, a, temp):
    # ------------------------
    """
    polynomial approximation for the partition function of various ions. 
    accurate to better than 4 percent for a temperature range
    from 3537° to 20160° K
    See Bolton, C.T., 1970, ApJ 161, 1187-88
    """
    theta = 5040 / temp
    ap_sum = 0.0
    for i in range(len(a)):
        ap_sum += a[i] * np.log(theta) ** i
    return g0 + np.exp(ap_sum)


# -------------------------------------------
def Bolton_partition_function(element, temp):
    # ---------------------------------------
    """
    returns the approximate partition function of ions
    contained in file partition_function_data.txt
    """
    f = prm.file_location + 'input_data/Bolton_partition_functions.txt'
    csv_reader = csv.reader(open(f), delimiter=' ')
    last_line_number = adat.row_count(f)

    for line in csv_reader:
        if csv_reader.line_num < last_line_number:
            a = []
            ion = str(line[0]).casefold()
            if ion == element.casefold():
                g0 = float(line[1])
                a_coef_number = int(line[2])
                for lx in range(a_coef_number):
                    a.append(float(line[3 + lx]))
                if 3500 < temp < 21000:
                    part_fcn = polyn_approx(g0, a, temp)
                else:
                    part_fcn = g0
                return part_fcn
            else:
                pass
        else:
            print('Warning - from Bolton partition function: ion not found')
            return 1.0


# -----------------------------------------
def Gray_partition_function(element, temp):
    # -------------------------------------
    """
    returns the approximate partition function of ions
    contained in file partition_function_data.txt
    Caution: temp is forced to the validity interval 2500<temp<25000.
    """
    f = prm.file_location + 'input_data/Gray_partition_functions.txt'
    csv_reader = csv.reader(open(f), delimiter=' ')
    last_line_number = adat.row_count(f)
    theta = np.linspace(0.2, 2.0, 10)
    th_min = theta[0]
    th_max = theta[-1]
    th = np.divide(5040, temp)
    th = np.where(th > th_min, th, th_min)
    th = np.where(th < th_max, th, th_max)
    for line in csv_reader:
        if line[0] == '#':
            pass
        elif csv_reader.line_num < last_line_number:
            pf = []
            ion = str(line[0]).casefold()
            # print(f"from Gray: ion = {ion}") if prm.lte_debug_mode else None
            if ion == element.casefold():
                atomic_number = float(line[1])
                ionization_level = float(line[2])
                print(f"ion = {ion}, atomic_number = {atomic_number}, ionization_level = {ionization_level}") \
                    if prm.lte_debug_mode else None
                for lx in range(10):
                    pf.append(float(line[3 + lx]))
                log_g0 = float(line[-1])
                log_part_fcn = np.interp(th, theta, pf)
                part_fcn = 10.0 ** log_part_fcn
                print(f"Grey partition function = {part_fcn}, log_g0 = {log_g0}") \
                    if prm.lte_debug_mode else None
                return part_fcn
            else:
                pass
        else:
            print(f"Warning - from Gray partition function: {element} not found")
            return 1.0


# ---------------------------------
def saha_phi(atom, eii, ei, temp):
    # -----------------------------
    """
    solves the temperature-dependent part phi(T) of the Saha equation 
    N1*ne/N0 = Phi(T) for 'atom';
    returns the LTE ratio of atoms N_ii in ionization state i+1 
    to atoms N_i in state i given the partition functions u1 = U(i), 
    u2 = U(i+1), the temperature temp in kelvins, the ionization energies eii
    and ei in eV of the higher and lower ionization levels.
    The result must be divided by the electron density ne in cm-3 to get N1/N0
    Caution: valid temperature range is 2500 < temp < 25000.
    Alternative form:
    I = eii - ei
    theta = 5040.0/temp
    phi = 1.2020e9 * (u1/u2) * theta**(-2.5) * 10**(-theta * I) / (k_B * temp)
    """
    print(f"entering saha_phi with atom = {atom}, eii = {eii}, ei = {ei}, temp = {temp}") \
        if prm.lte_debug_mode else None
    u1 = Gray_partition_function(atom + '_i', temp)
    u2 = Gray_partition_function(atom + '_ii', temp)
    phi = ((2.0 * u2) / u1) * (saha_cst * temp) ** 1.5 * np.exp(-(eii - ei) / (k * temp))
    print(f"saha_phi - u1 = {u1}, u2 = {u2}, ei = {ei}, eii = {eii}, phi = {phi}") \
        if prm.lte_debug_mode else None
    return phi


# ----------------------------------------
def saha_phi_prime(atom, eiii, eii, temp):
    # ------------------------------------
    """
    solves the temperature-dependent part phi(T) of the Saha equation
    N1*ne/N0 = Phi(T) for 'atom';
    returns the LTE ratio of atoms N_iii in ionization state i+2
    to atoms N_ii in state i+1 given the partition functions u2 = U(ii),
    u3 = U(iii), the temperature temp in kelvins, the ionization energies eii
    and ei in eV of the higher and lower ionization levels.
    The result must be divided by the electron density ne in cm-3 to get N1/N0
    Caution: valid temperature range is 2500 < temp < 25000.
    Alternative form:
    I = eiii - eii
    theta = 5040.0 / temp
    phi_prime = 1.2020e9 * (u2 / u3) * theta ** (-2.5) * 10 ** (-theta * I) / (k_B * temp)
    """
    print(f"entering saha_phi_prime with ion = {atom}, eiii = {eiii}, eii = {eii}, temp = {temp}") \
        if prm.lte_debug_mode else None
    u2 = Gray_partition_function(atom + '_ii', temp)
    u3 = Gray_partition_function(atom + '_iii', temp)
    phi_prime = ((2.0 * u3) / u2) * (saha_cst * temp) ** 1.5 * np.exp(-(eiii - eii) / (k * temp))
    print(f"saha_phi - u2 = {u2}, u3 = {u3}, eii = {eii}, eiii = {eiii}, phi_prime = {phi_prime}") \
        if prm.lte_debug_mode else None
    return phi_prime


# --------------------------------------------
def iter_loop(nt, el_nbr, ratio, na_over_nhs):
    # ----------------------------------------
    """
    iterate over the electron density ne
    """
    print(f"entering iter_loop - na_over_nhs \n {na_over_nhs}") \
        if prm.lte_debug_mode else None
    print(f"ratio \n {ratio}") if prm.lte_debug_mode else None
    ne = np.sqrt(nt)
    new_ne = ne
    niter = 0
    niter_max = 1000
    # num = np.zeros(el_nbr)
    # denum = np.zeros(el_nbr)
    # mu_denum = 0.0
    # pr1 = np.zeros(el_nbr)
    # pr2 = np.zeros(el_nbr)
    corr = 1.0
    while corr > prm.frac and niter < niter_max:
        niter += 1
        # sigma_top = 0.0
        # sigma_bottom = 0.0
        ne = new_ne
        print(f"iteration {iter} - ne = {ne}") if prm.lte_debug_mode else None
        num = [(ratio[iel] / ne) / (1.0 + (ratio[iel] / ne)) for iel in range(el_nbr)]
        pr1 = np.multiply(num, na_over_nhs)
        sigma_top = pr1.sum()
        if prm.lte_debug_mode:
            print(f"num \n {num}")
            print(f"np.shape(num) = {np.shape(num)}")
            print(f"na_over_nhs \n {na_over_nhs}")
            print(f"np.shape(na_over_nhs) = {np.shape(na_over_nhs)}")
            print(f"pr1 \n {pr1}")
            print(f"np.shape(pr1) = {np.shape(pr1)}")
            print(f"sigma_top = {sigma_top}")
        denum = np.add(num, 1)
        pr2 = np.multiply(denum, na_over_nhs)
        sigma_bottom = pr2.sum()
        new_ne = nt * np.divide(sigma_top, sigma_bottom)
        if prm.lte_debug_mode:
            print(f"pr2 \n {pr2}")
            print(f"np.shape(pr2) = {np.shape(pr2)}")
            print(f"sigma_bottom = {sigma_bottom}")
            print(f"new_ne = {new_ne}\n")

        corr = np.abs(new_ne - ne) / ne
        print(f"corr = {corr}") if prm.lte_debug_mode else None

    print(f"ne iteration ended after {niter} steps") if prm.lte_debug_mode else None
    if niter >= niter_max:
        print('no convergence in iter_loop for ne (LTE module)')
        exit()
    return ne


# -----------------------
def populations(nt, temp):
    # -------------------
    """
    returns the number ne of electrons per cm3 and the atomic abundances of the elements
    under investigation for temperature vector temp, as well as the populations of the first three
    ionization levels of these (except when the 3rd level partition functions are not given in Gray).
    We assume for the computation of ne an atomic gas made up of the mix 'elements'
    with total number density vector nt.
    Solar abundances and LTE conditions are also assumed for the population computations.
    The transcendental equation for ne is solved in a straightforward iterative way,
    in iter_loop, starting with ne = sqrt(nt).
    The mean molecular weight mu of the gas mixture considered is also returned. 
    NB. nt is the total number (ions + electrons) of particles per cm-3, so the gas pressure 
    is pg = nt * kT
    Caution: only the first ionization stage is included here. Slight departures over exact 
    result at temp < 5000K. The code was tested against Tabelle 4.7.3, Seite 154 in 
    Unsöld & Baschek's Der Neue Kosmos 4. Auflage
    """
    # read atomic data ordered in increasing weight
    element_list = np.array(['H', 'He', 'C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'K', 'Ca', 'Cr', 'Fe', 'Ni'])
    el_nbr = len(element_list)
    atoms = np.array(range(el_nbr), dtype='str')
    atomic_numbers = np.zeros(el_nbr)
    weights = np.zeros(el_nbr)
    g0s = np.zeros(el_nbr)
    g1s = np.zeros(el_nbr)
    log_a12s = np.zeros(el_nbr)
    na_over_nhs = np.zeros(el_nbr)
    ion_energ = np.zeros((el_nbr, 3))
    ion_wavel = np.zeros((el_nbr, 3))
    ratio_0 = np.zeros((el_nbr, prm.idr))
    # ratio_1 = np.zeros((el_nbr, prm.idr))
    # n0_over_nt = np.zeros((el_nbr, prm.idr))
    # n1_over_nt = np.zeros((el_nbr, prm.idr))
    # n2_over_nt = np.zeros((el_nbr, prm.idr))

    iel = -1
    # read all elements in element_list above to compute ne
    for element in element_list:
        iel += 1
        atoms[iel], atomic_numbers[iel], weights[iel], g0s[iel], g1s[iel], log_a12s[iel], na_over_nhs[iel], \
            ion_energ[iel, 0], ion_wavel[iel, 0], ion_energ[iel, 1], ion_wavel[iel, 1], \
            ion_energ[iel, 2], ion_wavel[iel, 2] = adat.read_atomic_data(element)
        if prm.lte_debug_mode:
            print(f"iel = {iel}, atoms[{iel}] = {atoms[iel]}")
            print(f"{element}, g0s[{iel}] = {g0s[iel]}, g1s[{iel}] = {g1s[iel]}, ion_energ = {ion_energ[iel, :]}, "
                  f"ion_wavel = {ion_wavel[iel, :]}")

        # compute Saha ratios 1-0
        for i in range(prm.idr):
            ratio_0[iel, i] = saha_phi(element, ion_energ[iel, 0], 0.0, temp[i])
        print(f"element = {element}, iel = {iel}, Saha ratio[{iel},:] \n{ratio_0[iel, :]}") \
            if prm.lte_debug_mode else None

    at_weight = weights * na_over_nhs
    at_metals = np.delete(at_weight, [0, 1])
    if prm.lte_debug_mode:
        print(f"at_weight \n{at_weight}")
        print(f"at_metals \n{at_metals}")

    # mean atomic weight
    wei = weights.sum()
    mu_num = at_weight.sum()
    mu_metals = at_metals.sum()
    mu_denum = na_over_nhs.sum()
    print(f"wei = {wei}, mu_num = {mu_num}, mu_metals = {mu_metals}, mu_denum = {mu_denum}") \
        if prm.lte_debug_mode else None

    ne = np.zeros(prm.idr)
    mu = np.zeros(prm.idr)

    for i in range(prm.idr):
        # iterate over electron density
        ne[i] = iter_loop(nt[i], el_nbr, ratio_0[:, i], na_over_nhs)
        # compute mean molecular weight
        mu[i] = at_weight.sum() / (na_over_nhs.sum() * (1.0 + ne[i] / nt[i]))
        print(f"ne/nt[{i}] = {ne[i] / nt[i]}, mu[{i}] = {mu[i]}") if prm.lte_debug_mode else None

    iel = -1
    # reset the arrays to zero
    atoms = np.array(range(el_nbr), dtype='str')
    atomic_numbers = np.zeros(el_nbr)
    weights = np.zeros(el_nbr)
    g0s = np.zeros(el_nbr)
    g1s = np.zeros(el_nbr)
    log_a12s = np.zeros(el_nbr)
    na_over_nhs = np.zeros(el_nbr)
    ion_energ = np.zeros((el_nbr, 3))
    ion_wavel = np.zeros((el_nbr, 3))
    ratio_0 = np.zeros((el_nbr, prm.idr))
    ratio_1 = np.zeros((el_nbr, prm.idr))
    n0_over_nt = np.zeros((el_nbr, prm.idr))
    n1_over_nt = np.zeros((el_nbr, prm.idr))
    n2_over_nt = np.zeros((el_nbr, prm.idr))

    # now compute the LTE level populations for the computed lines
    # recall that prm.elements are tuples (element_abbreviation, number_of_computed_lines_for_that_element)
    print('From populations') if prm.lte_debug_mode else None
    for elem, nlines in prm.elements:
        iel += 1
        atoms[iel], atomic_numbers[iel], weights[iel], g0s[iel], g1s[iel], log_a12s[iel], na_over_nhs[iel], \
            ion_energ[iel, 0], ion_wavel[iel, 0], ion_energ[iel, 1], ion_wavel[iel, 1], \
            ion_energ[iel, 2], ion_wavel[iel, 2] = adat.read_atomic_data(elem)
        print(atoms[iel], atomic_numbers[iel], weights[iel], g0s[iel], g1s[iel], log_a12s[iel], na_over_nhs[iel],
              ion_energ[iel, 0], ion_wavel[iel, 0], ion_energ[iel, 1], ion_wavel[iel, 1],
              ion_energ[iel, 2], ion_wavel[iel, 2]) if prm.lte_debug_mode else None
        if prm.lte_debug_mode:
            print(f"iel = {iel}, atoms[{iel}] = {atoms[iel]}")
            print(f"{elem}, g0s[{iel}] = {g0s[iel]}, g1s[{iel}] = {g1s[iel]}, ion_energ = {ion_energ[iel, :]}"
                  f", ion_wavel = {ion_wavel[iel, :]}")

    population_dict = {}
    ielement = -1

    for eleme, nlines in prm.elements:
        print(f"eleme = {eleme}, nlines = {nlines}") if prm.lte_debug_mode else None
        ielement += 1
        # compute Saha ratios
        for i in range(prm.idr):
            ratio_0[ielement, i] = saha_phi(eleme, ion_energ[ielement, 0], 0.0, temp[i]) / ne[i]
            n0_over_nt[ielement, i] = 1.0 / (ratio_0[ielement, i] + 1)

            if ion_energ[ielement, 1] != 0.0:
                ratio_1[ielement, i] = saha_phi_prime(eleme, ion_energ[ielement, 1], ion_energ[ielement, 0],
                                                      temp[i]) / ne[i]
                n0_over_nt[ielement, i] = 1.0 / (1 + ratio_0[ielement, i] * (1.0 + ratio_1[ielement, i]))  # OK
                n1_over_nt[ielement, i] = 1.0 / (1.0 / ratio_0[ielement, i] + ratio_1[ielement, i] + 1)  # OK
            else:
                n1_over_nt[ielement, i] = 1.0 - n0_over_nt[ielement, i]

        # the population dictionary contains the populations in the neutral ground state and first ionization level
        # ground state for each element. Provision is made for the population in the second ionization level
        # but is not used here.

        population_dict.update({eleme: [n0_over_nt[ielement, :], n1_over_nt[ielement, :],
                                        n2_over_nt[ielement, :]]})
    print(population_dict) if prm.lte_debug_mode else None

    return ne / nt, mu, g0s, g1s, population_dict


# -------------------------------------
def lte_tau(atom, ion_level, abund, lambda0, gl, gu, el_ev, eu_ev, flu, nt, temp):
    # ---------------------------------
    """
    returns the radial optical depths of the computed lines based on LTE level populations
    """
    # global atom, ion_level, a_weight, abund, lambda0, el_ev, eu_ev, gl, gu, flu, Aul, log_gf, ion_level_l, ion_level_u

    # use the Saha equation to compute equilibrium electron density and level population ratios
    ne, mu, g0s, g1s, population_dict = populations(nt, temp)
    print(f"Entering lte.tau for atom {atom} -- g0s = {g0s}, g1s = {g1s}") if prm.lte_debug_mode else None
    tau = np.zeros((prm.nline, prm.idr))
    nl = np.zeros((prm.nline, prm.idr))
    nu = np.zeros((prm.nline, prm.idr))
    nx_over_nt = np.zeros((prm.nline, 3), object)  # holds population dictionary above
    nl_over_nt = np.zeros(prm.nline, object)
    nu_over_nt = np.zeros(prm.nline, object)

    # use populations returned from abundances to compute the radial part taur of optical depth
    index = -1
    for element in population_dict.keys():
        # print(element)
        index += 1
        nx_over_nt[index, :] = population_dict.get(element)
        if prm.lte_debug_mode:
            print(f"el = {element}, lambda = {lambda0[index]}, prm.line_range[{index}] = {prm.line_range[index]}")
        nl_over_n0 = np.zeros(prm.idr, object)
        nu_over_nl = np.zeros(prm.idr, object)
        for lx in prm.line_range[index]:
            print(f"lx = {lx}, index = {index}, prm.line_range[{index}] = {prm.line_range[index]}") \
                if prm.lte_debug_mode else None
            g0 = g0s[index]
            print(f"g0s[{index}] = {g0s[index]}") if prm.lte_debug_mode else None
            print(f"g1s[{index}] = {g1s[index]}") if prm.lte_debug_mode else None
            if ion_level[lx] == 0:
                if g0 != 0:
                    print(f"{element} loop. lx = {lx}, index = {index}, g0 = {g0}") if prm.lte_debug_mode else None
                    nl_over_n0[lx] = boltzmann_ratio(gl[lx], g0, el_ev[lx], 0.0, temp)
                    nu_over_nl[lx] = boltzmann_ratio(gu[lx], gl[lx], eu_ev[lx], el_ev[lx], temp)
                    print(f"boltzmann ratio({gl[lx], g0, el_ev[lx]}, 0) \n{nl_over_n0[lx]}") \
                        if prm.lte_debug_mode else None
                    print(f"boltzmann ratio({gu[lx], gl[lx], eu_ev[lx], el_ev[lx]}) \n{nu_over_nl[lx]}") \
                        if prm.lte_debug_mode else None
                    nl_over_nt[lx] = nx_over_nt[index, 0] * nl_over_n0[lx]
                    nu_over_nt[lx] = nx_over_nt[index, 0] * nu_over_nl[lx] * nl_over_n0[lx]
                else:
                    print("element without g0 value. Abort.")
                    exit()
            else:
                print(f"ion_level[{lx}] = {ion_level[lx]}, element = {element}") if prm.lte_debug_mode else None
                # nl_over_n0 = 1.0 for resonance lines
                print(f"{element} loop. lx = {lx}, index = {index}, g0 = {g0}") if prm.lte_debug_mode else None
                nl_over_nt[lx] = nx_over_nt[index, 1]  # pop in first ionization state ground level
                nu_over_nt[lx] = boltzmann_ratio(gu[lx], gl[lx], eu_ev[lx], el_ev[lx], temp) * nx_over_nt[index, 1]
                print(f"boltzmann ratio({gu[lx], gl[lx], eu_ev[lx], el_ev[lx]}) \n{nu_over_nt[lx]}") \
                    if prm.lte_debug_mode else None
            if prm.lte_debug_mode:
                print(f"nl_over_nt[{lx}] \n{nl_over_nt[lx]}")
                print(f"nu_over_nt[{lx}] \n{nu_over_nt[lx]}")

            n_elem = nt * abund[lx]
            print(f"n_element \n{n_elem}") if prm.lte_debug_mode else None
            for i in range(prm.idr):
                nl[lx, i] = nl_over_nt[lx][i] * n_elem[i]
                nu[lx, i] = nu_over_nt[lx][i] * n_elem[i]
            if prm.lte_debug_mode:
                print(f"lx = {lx} \nnl[{lx}] \n {nl[lx, :]} \nnu[{lx}] \n{nu[lx, :]}")
                print(f"abund[{lx}] = {abund[lx]}")

            # tau is Eq. 4.7.29 in Ünsold & Bascheck's "Der Neue Kosmos" without the phi(nu) profile function.
            # m = u, n = l. We do not assume negligible stimulated emission in the following.
            tau[lx, :] = tau_cst * flu[lx] * nl[lx, :] * (1.0 - (nu[lx, :] / gu[lx]) / (nl[lx, :] / gl[lx]))

    return ne, nl, nu, mu, tau
