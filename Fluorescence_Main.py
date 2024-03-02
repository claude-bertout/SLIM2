"""
SLIM2 (Spectral Line Interactions in Moving Media) is a Python/Numpy code for solving the line formation problem
in 2D moving media, for multiple lines stemming from multiple elements, including the case of interacting lines
leading to fluorescence effects. The code, written by Claude Bertout, is open source under a CC-BY-SA 4.0 license.
See the A&A (future) publication and the README.md file for more details.

NB. The GitHub code is the version restricted to 3 interacting lines that has been used to produce the figures of
the publication. A more general version will be made available on GitHub in due time. The user should be able
to easily reproduce the paper figures by using the parameters given in the paper. The resulting plots and files,
including the parameter file, are also available in the subdirectory "results" for the user's convenience.

This module is, unsurprisingly, the main program, and it calls all other modules in turn to perform the computation.
"""


# ----------------
def main_module():
    # -------------

    # ----- import python libraries -----
    import numpy as np
    import time
    import sys
    import matplotlib.pyplot as plt
    from pathlib import Path

    # ----- import project modules -----
    import parameter_parser as parse
    import import_parameters as prm
    import set_grids as grid
    import import_line_data as ld
    import function_library as lib
    import source_module as sm
    import flux_integration_module as fli
    import plot_library as plres
    import in_out_library as io

    # print full arrays
    np.set_printoptions(threshold=sys.maxsize)

    # all input parameters are defined in computation_parameters.conf file and imported from prm
    # open logging file if in debugging mode
    orig_stdout = ''
    if prm.debug_mode or prm.silent_mode:
        orig_stdout = sys.stdout
        prefix_log = parse.config['log_file']
        out_file_log = prm.file_location_base + prefix_log + '.txt'
        print('Debugging or silent mode -- Output redirected to log file')
        sys.stdout = open(out_file_log, 'w')

    # define result files at base level: cp, crvs, ew depend only on geometry
    prefix_cp = parse.config['result_file_cp']
    out_file_cp = prm.file_location_base + prefix_cp + '.txt'
    prefix_plot_file_cp = parse.config['plot_file_cp']
    plot_file_cp = prm.file_location_base + prefix_plot_file_cp + '.png'
    prefix_plot_file_crvs = parse.config['plot_file_crvs']
    plot_file_crvs = prm.file_location_base + prefix_plot_file_crvs + '.png'
    prefix_ew = parse.config['result_file_ew']
    out_file_ew = prm.file_location_base + prefix_ew + '.txt'

    start_time = time.time()

    if prm.one_model:
        # single model
        int_min = 0  # define nt range
        int_max = 1
        itemp_min = 0  # define temp range
        itemp_max = 1
        nt0 = np.ones(prm.idr) * prm.ntc
        temp0 = np.ones(prm.idr) * prm.temp0

    else:
        # production run
        int0 = prm.int0
        itemp0 = prm.itemp0
        log_nt_min = np.log10(prm.nt_min)
        log_nt_max = np.log10(prm.nt_max)
        nt0 = np.logspace(log_nt_min, log_nt_max, int0)
        temp0 = np.linspace(prm.temp_min, prm.temp_max, itemp0)
        # divide computation in 4 series of 100 models each, with subsets [0:10] and [10:20] for each variable
        int_min = prm.i_nt_min  # define nt range
        int_max = prm.i_nt_max
        itemp_min = prm.i_temp_min  # define temp range
        itemp_max = prm.i_temp_max

    model_number = -1
    for nx in nt0[int_min:int_max]:
        for tx in temp0[itemp_min:itemp_max]:
            model_number += 1
            model_time = time.time()
            # Computation parameters for this loop
            ntx = nx * np.abs(grid.v_env[0]) * (prm.rc / grid.r) ** 2.0 / np.abs(grid.v_env)
            tempx = tx * np.ones(prm.idr)

            print("\n---------------------------------------------")
            print(f"Model {model_number} - nx = {nx:.2e}, tx = {tx:.2e}")
            print("---------------------------------------------\n")

            # result files are stored in results directory and stamped with date and time
            time_stamp = time.strftime('%d''-''%m''-''%Y''_''%H''%M''%S')
            Path(prm.file_location_base).mkdir(parents=True, exist_ok=True)
            if prm.one_model:
                file_location = prm.file_location_base + time_stamp + "/"
                Path(file_location).mkdir(parents=True, exist_ok=True)
            else:
                file_location = prm.file_location_base

            # data files
            prefix_prm = parse.config['result_file_prm']
            out_file_prm = file_location + prefix_prm + '.txt'
            prefix_source = parse.config['result_file_s']
            out_file_s = file_location + prefix_source + '.txt'
            prefix_profile = parse.config['result_file_p']
            out_file_p = file_location + prefix_profile + '.txt'
            prefix_blend = parse.config['result_file_b']
            out_file_b = file_location + prefix_blend + '.txt'

            # plot files
            prefix_plot_file_all = parse.config['plot_file_all']
            prefix_plot_file_input = parse.config['plot_file_input']
            plot_file_input = file_location + prefix_plot_file_input + '.png'
            prefix_plot_taur = parse.config['plot_file_taur']
            plot_file_taur = file_location + prefix_plot_taur + '.png'
            prefix_plot_file_s = parse.config['plot_file_s']
            plot_file_s = file_location + prefix_plot_file_s + '.png'
            prefix_plot_file_p = parse.config['plot_file_p']
            plot_file_p = file_location + prefix_plot_file_p + '.png'
            prefix_plot_file_b = parse.config['plot_file_b']
            plot_file_b = file_location + prefix_plot_file_b + '.png'

            # ---------------------------------
            # compute the line source functions
            # ---------------------------------

            # line ordering for the source function computation
            # depends on velocity field, see input_line_data module

            # define source function related arrays
            bnu = np.zeros((prm.nline, prm.idr))
            source = np.zeros((prm.nline, prm.idr))
            fc = np.zeros(prm.nline)
            epsilon = np.zeros((prm.nline, prm.idr))
            # delta_v_mat = np.zeros((prm.nline, prm.nline))
            # inter_cp = np.zeros((prm.nline, prm.nline), int)
            # self_cp = np.zeros((prm.nline, prm.nline), int)

            # define radial optical depth distribution taur = np.zeros((prm.nline, prm.idr))
            # assume lte populations
            # opticl depths are computed in order of increasing atomic weights
            # and reordered together with line properties according to velocity field
            lambda0, line_Doppler_width, ne, taur = ld.compute_optical_depths(ntx, tempx)

            # look for possible interactions between lines and earmark them
            if not model_number:
                delta_v_mat, i_vmax, self_cp, inter_cp = ld.compute_interaction_matrix(lambda0)

            nfreq, nfreq_tot, nwing, dwave, dvel, bxmax, dxd, f_dict = ld.define_frequency_grid(i_vmax,
                                                                                                line_Doppler_width)

            # compute the relevant line Planck functions and stellar background continuum
            for lx in range(prm.nline):
                bnu[lx, :] = lib.Planck_function(lambda0[lx], tempx[:])
                fc[lx] = lib.Planck_function(lambda0[lx], prm.teff)

            # compute the line collisional de-excitation rates
            # We use the old Van Regemorter approximation here.
            # More precise coefficients will be needed for comparison with observations

            if prm.epsilon_flag:
                epsilon = ld.collision_de_exc_rate(lambda0, tempx, ntx, ne)

            if prm.source_debug_mode:
                for lx in range(prm.nline):
                    print(f"epsilon[{lx}] \n {epsilon[lx]}")
                    print(f"fc[{lx}] \n {fc[lx]}")
                    print(f"bnu[{lx}] \n {bnu[lx]}")

            # -----------------------------
            # compute local source function
            # -----------------------------

            local_source, beta, betac = sm.compute_local_source_function(epsilon, bnu, fc, taur)

            if prm.source_debug_mode:
                for lx in range(prm.nline):
                    print(f"local source for lx = {lx} \n {local_source[lx, :]}")

            source_l = np.zeros((prm.nline, prm.idr))  # local source function vector
            # source_l is the normalized local source
            for lx in range(prm.nline):
                source_l[lx, :] = local_source[lx, :] / fc[lx]
                print(f"local source normalized to background continuum for lx = {lx} \n {source_l[lx, :]}") \
                    if prm.source_debug_mode else None

            # ---------------------------------------------------------------------
            # CAUTION - in case we do not compute the non-local part of the source
            # the normalized full source function vector is initialized to source_l
            # and the unnormalized full source term is initialized to local_source
            # ----------------------------------------------------------------------

            source_nl = source_l
            source = local_source

            # -----------------------------------------------------
            # compute non-local and/or interacting source functions
            # -----------------------------------------------------

            if prm.non_local or prm.interact_mode:
                print(f"Entering non-local source function computation...")
                # the cp-surfaces are computed once and stored for future use
                # define logical flags for controlling that
                if not model_number:
                    store_cp_flag = True
                    use_stored_cp_flag = False
                else:
                    store_cp_flag = False
                    use_stored_cp_flag = True

                source, final_iter = sm.compute_non_local_source_function(out_file_cp, store_cp_flag,
                                                                          use_stored_cp_flag,
                                                                          delta_v_mat, self_cp, inter_cp,
                                                                          epsilon, bnu, fc, beta, betac,
                                                                          local_source, taur)

                # normalize source_nl for plot purposes
                for lx in range(prm.nline):
                    source_nl[lx, :] = source[lx, :] / fc[lx]
                    if prm.source_debug_mode:
                        print(f"final iteration = {final_iter} - nl source normalized to background for lx = {lx}"
                              f"\n {source_nl[lx, :]}")

            if prm.graph_mode:
                plres.plot_sources(source_l, source_nl)

            # -------------------------------------
            # now perform the line flux integration
            # -------------------------------------

            core_emergent_flux = np.ones((prm.nline, nfreq_tot))
            env_emergent_flux = np.ones((prm.nline, nfreq_tot))
            # total_flux = np.zeros((prm.nline, nfreq_tot))
            # env_blend = np.zeros(nfreq_tot)
            # core_blend = np.zeros(nfreq_tot)

            print("Entering flux integration")  # if prm.flux_debug_mode else None

            # the flux integration is done in two steps: first the envelope (emission component)
            # then the stellar core (absorption component).
            # It is also possible to compute either one by itself for test purposes.
            # This is controlled by prm.env_only and prm.core_only
            if prm.env_only:
                print('Envelope region')
                region = 'envelope'
                pmin = prm.rc
                pmax = prm.rmax
                # set up p-grid for Gaussian integration
                xgp_env, wgp_env = np.polynomial.legendre.leggauss(prm.ide)
                p_env = 0.5 * (pmax - pmin) * xgp_env + 0.5 * (pmax + pmin)
                print(f"envelope p-grid \n {p_env}") if prm.debug_mode else None
                env_emergent_flux = fli.integrate_flux(region, prm.ide, nfreq, nfreq_tot, nwing, dxd, f_dict,
                                                       delta_v_mat, dvel, p_env, wgp_env, source, taur, fc)
                if prm.velocity_index == 1 or prm.velocity_index == 3:
                    # accretion flow. Flip velocities
                    for lx in range(prm.nline):
                        env_emergent_flux[lx] = np.flip(env_emergent_flux[lx])

            if prm.core_only:
                print('Core region')
                region = 'core'
                pmin = 1.0e-2 * prm.rc
                pmax = prm.rc
                # set up p-grid for Gaussian integration
                xgp_core, wgp_core = np.polynomial.legendre.leggauss(prm.idc)
                p_core = 0.5 * (pmax - pmin) * xgp_core + 0.5 * (pmax + pmin)
                print(f"core p-grid \n {p_core}") if prm.debug_mode else None
                core_emergent_flux = fli.integrate_flux(region, prm.idc, nfreq, nfreq_tot, nwing, dxd, f_dict,
                                                        delta_v_mat, dvel, p_core, wgp_core, source, taur, fc)
                # accretion flow. Flip velocities
                if prm.velocity_index == 1 or prm.velocity_index == 3:
                    for lx in range(prm.nline):
                        core_emergent_flux[lx] = np.flip(core_emergent_flux[lx])

            total_flux = core_emergent_flux + env_emergent_flux

            cpu_time = time.time() - model_time
            print(f"\nCPU time for this model = {cpu_time:.2f} seconds\n")

            # ----------------------------------
            # plot and store computation results
            # ----------------------------------
            # store line equivalent width information
            io.store_equivalent_widths(out_file_ew, lambda0, nfreq, nfreq_tot,
                                       dvel, ntx[0], tempx[0], total_flux, fc)
            if prm.one_model:
                # save sources
                io.save_sources(out_file_s, source_l, source_nl)
                # store emergent line profiles
                io.save_profiles(out_file_p, out_file_b, lambda0, nfreq, dvel, total_flux, fc)
                # store computation parameters
                io.save_parameters(out_file_prm, time_stamp, nx, tx, cpu_time)
                # plot velocity and density in the envelope
                plres.plot_v_and_nt()
                plt.savefig(plot_file_input, dpi=1200, bbox_inches='tight')
                plt.close()
                # plot line opacities and collisional de-excitation rates
                plres.plot_opacities(taur, epsilon)
                plt.savefig(plot_file_taur, dpi=1200, bbox_inches='tight')
                plt.close()
                # plot source functions
                plres.plot_nl_source(source_nl)
                plt.savefig(plot_file_s, dpi=1200, bbox_inches='tight')
                plt.close()
                # plot line profiles
                plres.plot_profiles(lambda0, nfreq, nfreq_tot, dvel, total_flux, fc)
                plt.savefig(plot_file_p, dpi=1200, bbox_inches='tight')
                plt.close()
                # plot both the fluorescent line and the overall line blend
                plres.plot_fluo_and_blend(lambda0, nfreq, nfreq_tot, dvel, total_flux, fc)
                plt.savefig(plot_file_b, dpi=1200, bbox_inches='tight')
                plt.close()
            # plot invariant geometric quantities for first model of the series
            if not model_number:
                plres.plot_3x3_crvs(delta_v_mat)
                plt.savefig(plot_file_crvs, dpi=1200, bbox_inches='tight')
                plt.close()
                if prm.interact_mode:
                    if not prm.velocity_index:
                        plres.read_and_plot_2_cp_surfaces(out_file_cp)
                        plt.savefig(plot_file_cp, dpi=1200, bbox_inches='tight')
                        plt.close()
                    else:
                        plres.read_and_plot_3_cp_surfaces(out_file_cp)
                        plt.savefig(plot_file_cp, dpi=1200, bbox_inches='tight')
                        plt.close()

            # close output file in debug mode
            if prm.debug_mode or prm.silent_mode:
                sys.stdout = orig_stdout

    total_cpu_time = time.time() - start_time
    print(f"Total execution time = {total_cpu_time:.2f} seconds")


if __name__ == '__main__':
    main_module()
