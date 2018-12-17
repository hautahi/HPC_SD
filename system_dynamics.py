#!/usr/bin/python3

"""
Thu 13 Dec 2018 09:04:02 AM PST

@author:    aaron heuser
@version:   3.1
@filename:  system_dynamics.py
"""

import csv
import datetime
import json
import sys
import system_dynamics_manager as system_dynamics_manager

if __name__ == '__main__':
    
    # Get the arguments passed in via the terminal. We look for one argument,
    # the path to a file with the run parameters, which is designated by the
    # flag '-f'.

    args = sys.argv[1:]
    run_fp = args[0]
    
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
