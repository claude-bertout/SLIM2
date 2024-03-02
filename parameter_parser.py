"""
modulename : parameter_parser
parses arguments in computation_parameters.conf and defines dictionaries
of all parameters to be read by input_parameters
"""
import argparse
from configparser import ConfigParser


# --------------------
def parse_arguments():
    # ----------------
    p = argparse.ArgumentParser(description='Spectral line transfer in moving stellar envelopes.')
    p.add_argument('-c', '--config_file', help='Path to config file.', default='computation_parameters.conf')
    p.add_argument('--nargs', nargs='+', type=str)
    return p.parse_args()


# ---------------------------------
def parse_config_file(config_file):
    # -----------------------------
    """
    CAUTION: upper case letters in parameter names raise KeyError exception in PyCharm !!
    """
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    default_config = {
        'project_directory': '/Users/claude/Documents/GitHub/TTS_Fluorescence/',
        'debug_mode': 'False',
        'silent_mode': 'False',
        'lte_debug_mode': 'False',
        'source_debug_mode': 'False',
        'flux_debug_mode': 'False',
        'graph_mode': 'False',
        'log_file': 'log',
        'result_file_p': 'profiles',
        'result_file_b': 'blend',
        'result_file_s': 'sources',
        'result_file_prm': 'parameters',
        'result_file_cp': 'cp-surfaces',
        'result_file_ew': 'ew-values',
        'plot_file_all': 'composite_plot',
        'plot_file_input': 'plot_file_input',
        'plot_file_taur': 'plot_file_taur',
        'plot_file_s': 'plot_file_s',
        'plot_file_p': 'plot_file_p',
        'plot_file_b': 'plot_file_b',
        'plot_file_cp': 'plot_file_cp',
        'plot_file_crvs': 'plot_file_crvs',
        'log_rgrid': 'False',
        'interact_mode': 'False',
        'sobolev_mode': 'False',
        'non_local': 'True',
        'epsilon_flag': 'False',
        'one_model': 'True',
        'velocity_index': '1',
        'core_only': 'False',
        'em_only': 'False',
        'idr': '50',
        'idz': '30',
        'idc': '24',
        'ide': '128',
        'rc': '1',
        'rmax': '10',
        'r_star': '2',
        'mc': '0.5',
        'alpha': '0.5',
        'teff': '3.5e3',
        'ntc': '1.0e12',
        'temp0': '8.0e3',
        'vturb0': '0.0',
        'frac': '1.0e-3',
        'eps': '1.0e-30',
        'conv': '1.0e-3',
        'iconv_max': '20',
        'nline': '2',
        'element': 'H',
        'elements': 'H',
        'nlines': '8',
        'ibeta': '4',
        'linewing': '2.0',
        'igauss_core': '6',
        'igauss_shell': '12',
        'int0': '20',
        'itemp0': '20',
        'nt_min': '3.0e14',
        'nt_max': '3.0e17',
        'temp_min': '4000.0',
        'temp_max': '14000.0',
        'i_nt_min': '0',
        'i_nt_max': '10',
        'i_temp_min': '0',
        'i_temp_max': '10'
    }
    cfg = ConfigParser(defaults=default_config)
    cfg.read(config_file)    
    dict_config = dict(cfg.items('main'))
    dict_config['project_directory'] = str(dict_config['project_directory'])
    dict_config['debug_mode'] = str2bool(dict_config['debug_mode'])
    dict_config['silent_mode'] = str2bool(dict_config['silent_mode'])
    dict_config['lte_debug_mode'] = str2bool(dict_config['lte_debug_mode'])
    dict_config['source_debug_mode'] = str2bool(dict_config['source_debug_mode'])
    dict_config['flux_debug_mode'] = str2bool(dict_config['flux_debug_mode'])
    dict_config['graph_mode'] = str2bool(dict_config['graph_mode'])
    # warning here, if any, is crazy Pycharm bug still there in 2023. Google it for details.
    dict_config['velocity_index'] = int(dict_config['velocity_index'])
    dict_config['log_rgrid'] = str2bool(dict_config['log_rgrid'])
    dict_config['non_local'] = str2bool(dict_config['non_local'])
    dict_config['interact_mode'] = str2bool(dict_config['interact_mode'])
    dict_config['epsilon_flag'] = str2bool(dict_config['epsilon_flag'])
    dict_config['sobolev_mode'] = str2bool(dict_config['sobolev_mode'])
    dict_config['log_file'] = dict_config['log_file']
    dict_config['result_file_prm'] = dict_config['result_file_prm']
    dict_config['result_file_cp'] = dict_config['result_file_cp']
    dict_config['result_file_ew'] = dict_config['result_file_ew']
    dict_config['result_file_s'] = dict_config['result_file_s']
    dict_config['result_file_p'] = dict_config['result_file_p']
    dict_config['result_file_b'] = dict_config['result_file_b']
    dict_config['plot_file_all'] = dict_config['plot_file_all']
    dict_config['plot_file_input'] = dict_config['plot_file_input']
    dict_config['plot_file_taur'] = dict_config['plot_file_taur']
    dict_config['plot_file_s'] = dict_config['plot_file_s']
    dict_config['plot_file_p'] = dict_config['plot_file_p']
    dict_config['plot_file_b'] = dict_config['plot_file_b']
    dict_config['plot_file_cp'] = dict_config['plot_file_cp']
    dict_config['plot_file_crvs'] = dict_config['plot_file_crvs']
    dict_config['idr'] = int(dict_config['idr'])
    dict_config['idz'] = int(dict_config['idz'])
    dict_config['idc'] = int(dict_config['idc'])
    dict_config['ide'] = int(dict_config['ide'])
    dict_config['rc'] = float(dict_config['rc'])
    dict_config['rmax'] = float(dict_config['rmax'])
    dict_config['r_star'] = float(dict_config['r_star'])
    dict_config['mc'] = float(dict_config['mc'])
    dict_config['alpha'] = float(dict_config['alpha'])
    dict_config['teff'] = float(dict_config['teff'])
    dict_config['ntc'] = float(dict_config['ntc'])
    dict_config['temp0'] = float(dict_config['temp0'])
    dict_config['vturb0'] = float(dict_config['vturb0'])
    dict_config['frac'] = float(dict_config['frac'])
    dict_config['eps'] = float(dict_config['eps'])
    dict_config['one_model'] = str2bool(dict_config['one_model'])
    dict_config['conv'] = float(dict_config['conv']) 
    dict_config['iconv_max'] = int(dict_config['iconv_max'])     
    dict_config['core_only'] = str2bool(dict_config['core_only'])
    dict_config['em_only'] = str2bool(dict_config['em_only'])
    dict_config['nline'] = int(dict_config['nline'])
    dict_config['element'] = str(dict_config['element'])
    dict_config['elements'] = str(dict_config['elements'])
    dict_config['nlines'] = str(dict_config['nlines'])
    dict_config['ibeta'] = float(dict_config['ibeta'])
    dict_config['linewing'] = float(dict_config['linewing'])
    dict_config['igauss_core'] = int(dict_config['igauss_core'])
    dict_config['igauss_shell'] = int(dict_config['igauss_shell'])

    # production run parameters
    dict_config['int0'] = int(dict_config['int0'])
    dict_config['itemp0'] = int(dict_config['itemp0'])
    dict_config['nt_min'] = float(dict_config['nt_min'])
    dict_config['nt_max'] = float(dict_config['nt_max'])
    dict_config['temp_min'] = float(dict_config['temp_min'])
    dict_config['temp_max'] = float(dict_config['temp_max'])
    # divide computation in 4 series of 100 models each, with subsets [0:10] and [10:20] for each variable
    # define nt range
    dict_config['i_nt_min'] = int(dict_config['i_nt_min'])
    dict_config['i_nt_max'] = int(dict_config['i_nt_max'])
    # define temp range
    dict_config['i_temp_min'] = int(dict_config['i_temp_min'])
    dict_config['i_temp_max'] = int(dict_config['i_temp_max'])

    return dict_config


args = parse_arguments()
config = parse_config_file(args.config_file)
