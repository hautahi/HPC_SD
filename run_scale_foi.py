"""
- This program runs the code for increasing number of "influence" parameters and records the running time.
"""

import json
import subprocess
import datetime
import pandas as pd
import os

param_file = "./data/run_parameters.json"
output_file = "./outputs/timings/deterministic_foi.csv"

for nfoi in [10,15,20,50,80,100]:
    
    print('*****')
    print("Running Simulation for FOI number: " + str(nfoi))
    print('*****')

    # open the run_parameters file
    with open(param_file, 'r') as f:
        params = json.load(f)
    
    # adjust years
    params['num_foi'] = nfoi
    params['years'] = 1
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
        pd.DataFrame({'num_foi' : [nfoi], 'time': [seconds]}).to_csv(output_file, index=False,mode='a',header=False)
    else:
        pd.DataFrame({'num_foi' : [nfoi], 'time': [seconds]}).to_csv(output_file, index=False)

