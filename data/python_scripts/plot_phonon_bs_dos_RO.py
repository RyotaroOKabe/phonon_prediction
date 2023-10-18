#%%
# Reads information from a DDB file, runs anaddb and plots
# the phonon band structure and DOS. Requires that abinit and the abipy
# python package are properly installed and configured.
# In particular a configuration manager.yaml file should be available.
# For further information see:
# http://abinit.github.io/abipy/workflows/taskmanager.html

import abipy.abilab as abilab

# Read DDB file
ddb = abilab.abiopen('/home/rokabe/data1/phonon/phonon_prediction/data/ddbs')

# Run anaddb to extract phonon frequencies and dos and read the results.
# Check docstrings to get a full description of all the options available
phbst, phdos = ddb.anaget_phbst_and_phdos_files()

# Plot bandstructure
phbst.plot_phbands()

# Plot phonon DOS
phdos.phdos.plot()

# Plot type-projected phonon DOS.
phdos.plot_pjdos_type()

# Plot the thermodynamic properties
phdos.phdos.plot_harmonic_thermo()


#%%

