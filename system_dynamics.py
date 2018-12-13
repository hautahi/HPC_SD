#!/usr/bin/python3

"""
Sun 09 Dec 2018 09:14:36 AM PST

@author:    aaron heuser
@version:   3.0
@filename:  system_dynamics.py
"""

import csv
import datetime
import json
import sys
import system_dynamics_manager

if __name__ == '__main__':
    
    # Get the arguments passed in via the terminal. We look for one argument,
    # the path to a file with the run parameters, which is designated by the
    # flag '-f'.
    
    args = sys.argv[1:]
    run_fp = args[args.index('-f') + 1]

    # Import the run parameters.
    with open(run_fp, 'r') as f:
        params = json.load(f)
    print('Run parameters imported. Now determining system solution.')
    start = datetime.datetime.now()
    sdm = system_dynamics_manager.SystemDynamicsManager(params)
    seconds = (datetime.datetime.now() - start).total_seconds()
    print('')
    print('*****')
    print('System solution determined in %f seconds.' % seconds)
    print('Output files can be found in the folder \'./output\'.')
    print('*****')
