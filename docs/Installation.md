Installation
============

### 1. Operating System (OS)

The project is established, tested and documented for a linux distribution. Specifically, the project
is developed under `Debian GNU/Linux 18.04 (stretch) x86_64` operating system.

### 2. Programming Language

Project is developed in [Python](https://www.python.org/) - version `3.6+`.
is developed under `Debian GNU/Linux 9.5 (stretch) x86_64` operating system.

### 3. Prerequisites

Please install the necessary, system-wide dependencies below: 

~~~
$ sudo apt install python3-tk
$ sudo apt install libpq-dev python3-dev
~~~

### 3. Install Cuda & cudnn

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64  

# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb  
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb  
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub  
sudo apt-get update  
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb  
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb  
sudo apt-get update  

# Install NVIDIA driver
sudo apt-get install --no-install-recommends nvidia-driver-418
# Reboot. Check that GPUs are visible using the command: nvidia-smi

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \  
    cuda-10-0 \  
    libcudnn7=7.6.2.24-1+cuda10.0  \  
    libcudnn7-dev=7.6.2.24-1+cuda10.0  


# Install TensorRT. Requires that libcudnn7 is installed above.
sudo apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda10.0 \  
    libnvinfer-dev=5.1.5-1+cuda10.0  

#### 4. Getting started

1. Create a folder named capstone-project-2019 and clone the project inside it  
2. Set the folder named capstone-project-2019 as your working directory
3. Create a new clean environment manually and install the libraries listed in the 
[requirements section](Requirements.md).  
Alternatively, open the terminal, go to the working directory and run the 
command: make -f. It will automatically create a new environment named thesis2019 in 
which the required libraries will be installed.  
4. Place a dataset named dataset.py in the /data/dataset directory. *  
5. Run the file named crawler.py. *  

*More information about 4th and 5th steps can be found in the [Data Acquisition section](Data_acquisition.md)

