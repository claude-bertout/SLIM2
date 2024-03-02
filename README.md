SLIM2 (Spectral Line Interactions in Moving Media) is a Python/Numpy code for solving the line formation problem
in 2D moving media, for multiple lines stemming from multiple elements, in the special case of interacting
lines leading to fluorescence effects. In the framework of a well-tested hybrid approach to the problem of line
formation, it first computes the generalized Sobolev source functions and then integrates exactly the line flux.
The lines considered for illustrative purposes participate in the FeI fluorescence observed in some T Tauri
stars. The atomic level populations are LTE so that the line optical depths are considered constant.

A future version of the code will allow iterating between radiation field and rate equations to obtain
the nlte level populations.

The code makes extensive use of the NumPy ndarray((dim1, dim2), object) construct, which is particularly useful
for ray-tracing procedures, although it is frowned upon by some pythonistas because if complicatess
array broadcasting. Please refer to A&A publication Bertout 2024 for the theoretical background leading to the
equations implemented here and read below for a description of the implementation.

Caveat: This is a research code and thus in constant flux. Version 1.0 presented here is stable, reasonably robust,
and optimized for solving the problem of line formation for the 3-line blend around 3969 Angstrom in TTS spectra. 
Plotting routines are restricted to that case. They are included in the project so that prospective users will be 
able to reproduce easily the A&A publication figures and make sure the code is working properly.

A more general version of the code, able to compute many interacting and non-interacting lines simultaneously
and allowing for variable temperature in the envelope, is currently being tested with the aim to utimately solve
the multi level problem.

Disclaimer. There is no guarantee whatsoever that this code is bugfree or that the theory underlying it is correct. 
Users do so at their own risk and cannot hold the author responsible for any disagreement that would possibly result 
from code errors or theory mistakes. Errare humanum est.

The code is publicly available under the Creative Commons 4.0 BY-NC-SA license. 
See the Creative Commons site for details.

Dependencies

The program requires the following python packages
- scipy
- numpy
- matplotlib
- argparse
- configparser

Code structure

-- "Fluorescence_Main.py" contains a single function "main_module" that calls all other modules 
to perform the computation. Its results are stored in a sub-directory of "results" directory 
marked with a date-time stamp of the start of computation.

The program involves three main steps:

>>> First step: computation set up

-- "import_parameters.py" imports all input parameters from "computation_parameters.conf". For this, an ancillary module
"parameter_parser.py" is needed. Several different options are possible for the computation (local or non-local 
Sobolev source, non-interacting or interacting lines, Sobolev or exact flux mode, single model or grid of models, 
various velocity fields, several levels of logging,  etc.) so that functional parameters are numerous. There are also 
a few physical parameters, such as the maximum density, envelope temperature, and stellar properties that must be 
defined. The meaning of each of these parameters is explained in the .conf file and default values are indicated.

-- "set_grids.py" sets up the spatial, velocity, and density grids, and analyzes the velocity field structure.

-- "import_line_data.py" reads the line data from "import_atomic_data.py" and computes and stores as arrays the line 
optical depths using "LTE_module.py". It also computes the absorption line profiles, defines the line interaction 
matrices, and sets up the overall frequency grid for the flux integration. Databases for partition functions and 
abundances must be stored in a directory named "input_data" containing the following files: abundances.txt, atomic_data.txt,
Bolton_partition_functions.txt, Gray_partition_functions.txt, partition_function_data.txt.

>>> Second step: number crunching

-- "source_module.py" first compute the local source functions, then finds and orders all the common point 
(CP) surfaces discussed above on high-resolution angular grids. 
This step is a major CPU-time consumer, and we store the CP-surface geometries for repeated use 
when a string of models with the same geometrical properties are calculated. All integrals over solid angles 
that appear in the computation of the source function are solved by Gaussian quadratures. The local source 
functions are iterated with the non-local contributions until convergence occurs. Although there 
are often more than two resonant surfaces on each line of sight, convergence occurs relatively rapidly for the 
velocity fields considered above and the many others that we tested. The Rybicki and Hummer (1978) theorem about 
convergence in the case of two interacting surfaces could probably be generalized to multiple surfaces, at least 
in the LTE case that we considered here.

-- "function_library.py" contains ancillary functions for "source_module.py" and "flux_integration.py".

-- "constants.py" contains some physical constants of relevance.

-- "flux_integration_module.py" integrates the line intensities over the impact parameter grid. A z-grid is defined 
on each impact parameter, and is transformed by interpolation into a grid of line frequencies present on the 
given ray. Contributing frequencies to each line are then found by taking into account their velocity 
displacements and resonant regions are deduced from there. Incremental line intensities are then computed 
and the flux follows from integrating the emergent intensities.

>>> Third step: storing and imaging results

-- in_out_library contains functions for storing the results and reading them for future use 
in a sub-directory of "results" (see above).

-- Finally, "plot_library" contains plot functions for various quantities plotted when prm.graph_mode = True,
as well as for plotting computation results.
	

