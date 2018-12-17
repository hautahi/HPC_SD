"""
- This program runs the code for increasing number of years.
"""

import json
import subprocess
import datetime
import pandas as pd
import os

param_file = "./data/run_parameters.json"
output_file = "./outputs/timings/deterministic_years.csv"

for year in [30,50]:
    
    print('*****')
    print("Running Simulation for years: " + str(year))
    print('*****')

    # open the run_parameters file
    with open(param_file, 'r') as f:
        params = json.load(f)
    
    # adjust years
    params['years'] = year
    params['stochastic'] = 0

    # save adjusted parameters file
    with open(param_file, 'w') as f:
        json.dump(params, f)
    
    # run simulation
    start = datetime.datetime.now()
    subprocess.call(["python3","system_dynamics.py","./data/run_parameters.json"], shell=False)
    seconds = (datetime.datetime.now() - start).total_seconds()

    # Save time to file
    if os.path.isfile(output_file):
        pd.DataFrame({'year' : [year], 'time': [seconds]}).to_csv(output_file, index=False,mode='a',header=False)
    else:
        pd.DataFrame({'year' : [year], 'time': [seconds]}).to_csv(output_file, index=False)

