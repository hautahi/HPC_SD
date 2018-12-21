# Parallel Calculations

This directory contains:

 - Shell scripts and Python scripts for running jobs on EC2
    - Also Python template scripts that are used to dynamically create run scripts like Hautahi's
 - A setup script for AWS EC2 instances on Amazon Linux
 - A Jupyter notebook for plotting and some CSV exports
    - Plots
    - An R script for analyzing the scaling that uses a Jupyter-generated CSV

I also have another directory, `outputs/parallel_timings`, with the timings.

 - Most timings were run sequentially on a single EC2 instance
 - Some timings (stochastic `mcsims` and `years`, mainly) were run singly on many AWS instances to get them done in time


## AWS Process

1. `scp` the Python source, `data` directory, and `parallel` directory to AWS.
2. `bash parallel/setup-aws`
3. Launch `screen` for persistence
4. `mkdir 72-cores` or similar
5. `python3 parallel/update-scaling-files.py -o [dir] -n [cores] -t parallel`
    - **Add the `-s` flag for stochastic simulations**
    - This uses the `.tmpl` files in `parallel` to generate new `run_scale` scripts
    - The years scaling must be run last because I didn't update the other scripts for `steps`
6. `bash parallel/execute.sh` to run all simulations as configured (execute the `run_scale` scripts)
7. `scp` the results back down


## Parallel  Code

My parallel implementation is `system_dynamics_manager_parallel.py`. I've also modified `system_dynamics.py` to choose the parallel implementation whenever the number of cores (`-n`) is different from 1.
