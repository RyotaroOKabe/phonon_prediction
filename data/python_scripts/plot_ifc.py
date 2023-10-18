# Reads information from a DDB file, runs anaddb and plots
# the interatomic force constants in real space. Requires that abinit
# and the abipy python package are properly installed and configured.
# In particular a configuration manager.yaml file should be available.
# For further information see:
# http://abinit.github.io/abipy/workflows/taskmanager.html

import abipy.abilab as abilab

# Read DDB file
ddb = abilab.abiopen('/home/rokabe/data1/phonon/phonon_prediction/data/ddbs')

# Run anaddb to extract the interatomic forces and read the results.
# Check docstrings to get a full description of all the options available
ifc = ddb.anaget_ifc()

# Plot the longitudinal interatomic force constants
ifc.plot_longitudinal_ifc()
