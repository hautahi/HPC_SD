"""
- This program runs the code for increasing number of addiction levels and records the runtime
- Make sure to change "output_file" & "stochastic" variable to switch between deterministic & stochastic
"""

import json
import subprocess
import datetime
import pandas as pd
import os
import csv

param_file = "./data/run_parameters.json"
fname = './data/stock_counts.csv'
output_file = "./outputs/timings/stochastic_addiction.csv"
stochastic = 1

for addiction in [4,5,10,20,30,40,50]:
    
    print('*****')
    print("Running Simulation for addiction: " + str(addiction))
    print('*****')

    # open the run_parameters file
    with open(param_file, 'r') as f:
        params = json.load(f)
    
    # adjust years
    params['years'] = 1
    params['num_foi'] = 10
    params['stochastic'] = stochastic
    params['sruns'] = 10

    # save adjusted parameters file
    with open(param_file, 'w') as f:
        json.dump(params, f)

    # open the stock count file and adjust
    with open(fname) as inf:
        reader = csv.reader(inf.readlines())
    
    lines = list(reader)
    # set number of age groups
    lines[-1][1] = str(3)
    # set number of addiction levels
    lines[-2][1] = str(addiction)

    with open(fname, 'w') as outf:
        writer = csv.writer(outf)
        for line in lines:
            writer.writerow(line)
    
    # run simulation
    start = datetime.datetime.now()
    subprocess.call(["python3","system_dynamics.py","./data/run_parameters.json"], shell=False)
    seconds = (datetime.datetime.now() - start).total_seconds()

    # Save time to file
    if os.path.isfile(output_file):
        pd.DataFrame({'addiction' : [addiction], 'time': [seconds]}).to_csv(output_file, index=False,mode='a',header=False)
    else:
        pd.DataFrame({'addiction' : [addiction], 'time': [seconds]}).to_csv(output_file, index=False)

