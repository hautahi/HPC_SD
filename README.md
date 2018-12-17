# High Performance Computing Implementations of System Dynamics Algorithms
Python code developed by Aaron to implement IMPAQ System Dynamics Model on AWS

## File Description

- The `data` directory contains the input parameter files.
- `system_dynamics_manager.py` defines all the necessary functions to run the algorithms.
- `system_dynamics.py` calls the functions defined above to solve the SD model. It can be run from the command line by 

  `python3 system_dynamics.py -f ./data/run_parameters.json`
  

## AWS Multicore Instructions
1. Setup instance on the AWS website
2. Login to AWS instance via: `ssh -i path/to/amazonkey.pem ec2-user@instance-address.amazonaws.com`
3. Setup AWS instance with: `sudo yum install python3`, `sudo yum install git` and `sudo pip3 install numpy pandas joblib scipy`
4. Transfer file to instance: `scp -i amazonkey.pem file_name ec2-user@instance-address.amazonaws.com:`
5. Transfer folder to instance: `scp -i amazonkey.pem -r folder_name ec2-user@instance-address.amazonaws.com:`
6. Transfer files back to local machine: `scp -i amazonkey.pem -r ec2-user@instance-address.amazonaws.com: .`
