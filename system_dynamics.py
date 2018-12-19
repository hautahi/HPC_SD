#!/usr/bin/python3

"""
Thu 13 Dec 2018 09:04:02 AM PST

@author:    aaron heuser
@version:   3.1
@filename:  system_dynamics.py
"""

import argparse
import csv
import datetime
import json
import sys
import system_dynamics_manager as manager
import system_dynamics_manager_parallel as parallel_manager

if __name__ == '__main__':
    
    # Get the arguments passed in via the terminal.
    # One required argument: the path to a file with the run parameters.
    # One optional argument -n, the number of cores. -n 1 => serial version.
    ap = argparse.ArgumentParser()
    ap.add_argument('params', metavar='<parameter file>', type=str)
    ap.add_argument(
        '-n', metavar='cores', type=int, default=1,
        help='Number of cores to use in the parallelization [default: 1]'
    )
    ap.add_argument(
        '-s', action='store_true',
        help='Set a seed for reproducibility'
    )
    args = ap.parse_args()
    
    # Import the run parameters.
    run_fp = args.params
    with open(run_fp, 'r') as f:
        params = json.load(f)
    print('Run parameters imported. Now determining system solution.')
    
    # Set a random seed if requested
    if args.s:
        RANDOM_SEED = 2**10
        print('Setting random seed: %d' % RANDOM_SEED)
        import random
        import numpy as np
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
    
    start = datetime.datetime.now()
    if args.n == 1:
        print('Executing serial version')
        sdm = manager.SystemDynamicsManager(params)
    else:
        print('Executing parallel version with %d cores' % args.n)
        sdm = parallel_manager.SystemDynamicsManager(params, ncores=args.n)
    seconds = (datetime.datetime.now() - start).total_seconds()
    print('')
    print('*****')
    print('System solution determined in %f seconds.' % seconds)
    print('Output files can be found in the folder \'./output\'.')
    print('*****')
