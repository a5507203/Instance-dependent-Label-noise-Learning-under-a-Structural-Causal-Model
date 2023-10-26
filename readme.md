Code for NeuripsIPS 2021 Paper "Instance-dependent Label-noise Learning under a Structural Causal Model" 

<div align="center">   
  
# Instance-dependent Label-noise Learning under a Structural Causal Model
[![Paper](https://img.shields.io/badge/NeuripsIPS21-green)](https://proceedings.neurips.cc/paper/2021/file/23451391cd1399019fa0421129066bc6-Paper.pdf3)

</div>

ARXIV: 
https://arxiv.org/pdf/2109.02986.pdf


Abstract: 
Label noise generally degenerates the performance of deep learning algorithms because deep neural networks easily overfit label errors. Let X and Y denote the instance and clean label, respectively. When Y is a cause of X, according to which many datasets have been constructed, e.g., SVHN and CIFAR, the distributions of $P(X)$ and (Y|X) are generally entangled. This means that the unsupervised instances are helpful in learning the classifier and thus reduce the side effects of label noise. However, it remains elusive on how to exploit the causal information to handle the label-noise problem. We propose to model and make use of the causal process in order to correct the label-noise effect. Empirically, the proposed method outperforms all state-of-the-art methods on both synthetic and real-world label-noise datasets.

Note that this is the implementation without using data augmentation. Please refer to the branch "new-implementation" for the source code with using the data augmentation.

+++++environment configuration++++++

#########important###############

The code is only tested on Linux Linux-based system (Ubuntu 20.04). 
The Python version is 3.6.9. The Pytorh version is 1.2.0 with GPU acceleration. 

It is unknown if the code is compatible on windows or different versions of Pytorh and Python. 
We have not tested to run our code in a CPU environment. 


+++++run experiments++++++

We provide shell scripts that allow to run all experiments with several lines of commands. 

#########run synthetic noise experiments on a specific dataset with a fixed sample size################

Open a terminal at the project root directory and type the following command to run CIFAR10, CIFAR100, SVHN, and F-MNSIT:

sudo chmod 755 syn_noise.sh
./syn_noise.sh


