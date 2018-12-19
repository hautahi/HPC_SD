# High Performance Computing Implementations of System Dynamics Algorithms
Python code developed by Aaron to implement IMPAQ's System Dynamics Model on AWS

## Folders

- The `data` directory contains the input parameter files.
- The `outputs` directory contains the results from the various model runs, which includes simulated model output, timing of runs and plots.

## Main file description

- `system_dynamics_manager.py` defines all the necessary functions to run the algorithms.
- `system_dynamics_manager_parallel.py` is mostly a copy of `system_dynamics_manager.py` (boo), but with the option for full parallelization of the calculation.
- `system_dynamics.py` calls the functions defined above to solve the SD model. It can be run from the command line by 

  `python3 system_dynamics.py ./data/run_parameters.json`
  `python3 system_dynamics.py -n [# cores] ./data/run_parameters.json`
  `python3 system_dynamics.py -s ./data/run_parameters.json`

## Other file description

- the `run_scale_x.py` files run the model by scaling up dimension x. The file must be edited at each run to switch between the stochastic and deterministic model by changing the `stochastic` and `output_file` parameters in the script.

## AWS Multicore Instructions
1. Setup instance on the AWS website
2. Login to AWS instance via: `ssh -i path/to/amazonkey.pem ec2-user@instance-address.amazonaws.com`
3. Setup AWS instance with: `sudo yum install python3 git tmux` and `sudo pip3 install numpy pandas joblib scipy`
4. Transfer file to instance: `scp -i amazonkey.pem file_name ec2-user@instance-address.amazonaws.com:`
5. Transfer folder to instance: `scp -i amazonkey.pem -r folder_name ec2-user@instance-address.amazonaws.com:`
6. Transfer files back to local machine: `scp -i amazonkey.pem -r ec2-user@instance-address.amazonaws.com: .`
7. Tip: Use `tmux` command before running a script to open a new screen. Transition back to main screen with `ctrl+b,d` and then back again using `tmux attach -d`. This allows you to log out of AWS while keeping a script running.
