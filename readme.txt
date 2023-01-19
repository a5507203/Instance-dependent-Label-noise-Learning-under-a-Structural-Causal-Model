Code for NIPS2021 Paper "Instance-dependent Label-noise Learning under a Structural Causal Model" (https://arxiv.org/pdf/2109.02986.pdf)
+++++environment configuration++++++

#########important###############
The code is only tested on Linux based System (Ubuntu 20.04). 
The python version is 3.6.9. The pytorh version is 1.2.0 with GPU acceleration. 

It is unknown that if the code is compatible on windows or different versions of pytorh and python. 
We have not tested to run our code in a CPU environment. 

Upon acception of the paper, we will test the compatibility of the code under different environments and publish the code on GitHub.
To avoid errors caused by inconsistent environment, you are encouraged to run our code under a same environment.
#################################


+++++run experiments++++++
We provide shell scripts that allows to run all experiments with several lines of commands. 

#########run synthetic noise experiments on a specific dataset with the fixed sample size################
Open a terminal at the project root directory and type the following command to run CIFAR10,CIFAR100,SVHN and F-MNSIT:

sudo chmod 755 syn_noise.sh
./syn_noise.sh


