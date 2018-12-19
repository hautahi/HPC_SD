#!/usr/bin/env bash

python3 run_scale_addiction.py 2>&1 >> logfile
python3 run_scale_age.py 2>&1 >> logfile
python3 run_scale_foi.py 2>&1 >> logfile
python3 run_scale_mcsims.py 2>&1 >> logfile
python3 run_scale_years.py 2>&1 >> logfile
