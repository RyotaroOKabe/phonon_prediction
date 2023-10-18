# Reads information from a DDB file, runs anaddb and extracts
# the values of the Born effective charges (BECs) and macroscopic
# dielectric tensor. Requires that abinit and the abipy python
#  package are properly installed and configured.
# In particular a configuration manager.yaml file should be available.
# For further information see:
# http://abinit.github.io/abipy/workflows/taskmanager.html

import abipy.abilab as abilab

# Read DDB file
ddb = abilab.abiopen('/home/rokabe/data1/phonon/phonon_prediction/data/ddbs')

# Run anaddb to extract BECs and dielectric tensor and read the results.
# Check docstrings to get a full description of all the options available
emacro, becs = ddb.anaget_emacro_and_becs()

print(emacro)

print(becs)