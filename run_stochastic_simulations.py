"""
- This program runs the stochastic code many times and saves the results into a separate folder.
- Make sure to set the relevant stochastic parameter toggle in the run_parameters.json file.
"""

import subprocess

for sim in range(100):
    
    print('*****')
    print("Running Simulation " + str(sim))
    print('*****')

    # run stochastic simulation
    subprocess.call(["python3","system_dynamics.py","./data/run_parameters.json"], shell=False)
    
    # move output to folder
    subprocess.call(["mv","output","./outputs/stochastic_simulation_output/output_" + str(sim)], shell=False)
