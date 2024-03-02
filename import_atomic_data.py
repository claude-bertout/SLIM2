"""
Module name: import_atomic_data
This module imports the atomic data necessary to feed the line_data module.
The data are read from atomic_dat.txt
"""
import import_parameters as prm
import function_library as lib
import numpy as np
import csv
from constants import angstrom


# ----------------------
def row_count(filename):
    # ------------------
    """
    returns the line number of file with name filename
    """
    with open(filename) as in_file:
        return sum(1 for _ in in_file)


# --------------------------------------------------------------
def read_line_data(element, nline, line_range, line_properties):
    # ----------------------------------------------------------
    """
    read relevant atomic data for the lines under consideration
    note tha line_properties is passed as a parameter rather than returned
    so that the full line list can build up over several passes in the subroutine.
    Caution: elements are sorted by atomic weight in the table of atomic data
    and must remained in that order when computing the optical depths.
    """
    print(f"Read line data")
    print(f"Atom = {element}, number of lines = {nline}, line_range = {line_range}")

    f = prm.file_location + 'input_data/atomic_data.txt'
    csv_reader = csv.reader(open(f), delimiter=' ')
    last_line_number = row_count(f)
    iel = line_range[0] - 1
    atom = ''

    for line in csv_reader:
        if line == [] or line[0] == '#':
            pass
        elif csv_reader.line_num <= last_line_number:
            atom = str(line[0]).casefold()
            if atom == element.casefold():
                iel += 1
                ion_level = int(line[1])
                a_weight = float(line[2])
                abund = float(line[3])
                lambda1 = float(line[4])
                # use air wavelengths for consistency with older optical spectroscopy work
                if str(line[5]) == 'vac':
                    print(f"Oops, it seems that we need to implement a vac2air routine "
                          f"to get all line wavelengths on the same scale. Abort at this time")
                    exit()
                else:
                    lambda0 = lambda1 * angstrom
                El_eV = float(line[6])  # lower electronic level
                Eu_eV = float(line[7])  # upper electronic level
                gl = float(line[8])
                gu = float(line[9])
                Aul = float(line[10])
                flu = float(line[11])
                log_gf = float(line[12])
                ion_level_l = float(line[13])  # lower ionization energy (ground state = 0.0)
                ion_level_u = float(line[14])  # upper ionization energy in eV
                line_properties.append([atom, ion_level, a_weight, abund, lambda0, El_eV, Eu_eV,
                                        gl, gu, Aul, flu, log_gf, ion_level_l, ion_level_u])

                print(f"iel = {iel}, atom = {atom}, ion_level = {ion_level}, a_weight = {a_weight}, "
                      f"abund = {abund}, lambda0 = {lambda0 / angstrom:.2f}, El_eV = {El_eV:.4f}, Eu_eV = {Eu_eV:.4f}\n"
                      f"flu = {flu:.4e}, gl = {gl}, gu = {gu}, Aul = {Aul:.4e}, log_gf = {log_gf:.4f}, "
                      f"ion_level_l = {ion_level_l}, ion_level_u = {ion_level_u}")
                if iel == line_range[-1]:
                    break

            else:
                pass
        else:
            print(f"from read_line_data: {atom} lines not found")
            exit()
    print('exit read line data')


# ----------------------------
def read_atomic_data(element):
    # ------------------------
    """
    returns the abundances, ionization energies and wavelengths for selected element
    from file abundances.txt. Used for computing the LTE level occupation numbers
    """
    f = prm.file_location + 'input_data/abundances.txt'

    csv_reader = csv.reader(open(f), delimiter=' ')
    last_line_number = row_count(f)
    iel = -1
    el_nbr = -1
    atom = ''
    ion_energ = np.zeros((prm.nelem, 3))
    ion_wavel = np.zeros((prm.nelem, 3))
    for line in csv_reader:
        if line[0] == '#':
            pass
        elif csv_reader.line_num < last_line_number:
            iel += 1
            atom = str(line[1]).casefold()
            if atom == element.casefold():
                el_nbr += 1
                atomic_number = int(line[0])
                weight = float(line[2])
                g0 = int(line[3])
                g1 = int(line[4])
                log_A12 = float(line[5])
                na_over_nh = float(line[6])
                ion_energ[el_nbr, 0] = float(line[7])
                ion_wavel[el_nbr, 0] = float(line[8])
                ion_energ[el_nbr, 1] = float(line[9]) if float(line[9]) > 0.0 else 0.0
                ion_wavel[el_nbr, 1] = float(line[10]) if float(line[10]) > 0.0 else 0.0
                ion_energ[el_nbr, 2] = float(line[11]) if float(line[11]) > 0.0 else 0.0
                ion_wavel[el_nbr, 2] = float(line[12]) if float(line[12]) > 0.0 else 0.0

                return (atom, atomic_number, weight, g0, g1, log_A12, na_over_nh, ion_energ[el_nbr, 0],
                        ion_wavel[el_nbr, 0], ion_energ[el_nbr, 1], ion_wavel[el_nbr, 1], ion_energ[el_nbr, 2],
                        ion_wavel[el_nbr, 2])
            else:
                pass
        else:
            print(f"from abundances: {atom} element not found. Aborting.")
            exit()
