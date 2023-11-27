Code for NeurIPS 2021 Paper "Instance-dependent Label-noise Learning under a Structural Causal Model" 

<div align="center">   
  
# Instance-dependent Label-noise Learning under a Structural Causal Model
[![Paper](https://img.shields.io/badge/NeuripsIPS21-green)](https://proceedings.neurips.cc/paper/2021/file/23451391cd1399019fa0421129066bc6-Paper.pdf3)

</div>

ARXIV: 
https://arxiv.org/pdf/2109.02986.pdf


Abstract: 
Label noise generally degenerates the performance of deep learning algorithms because deep neural networks easily overfit label errors. Let X and Y denote the instance and clean label, respectively. When Y is a cause of X, according to which many datasets have been constructed, e.g., SVHN and CIFAR, the distributions of $P(X)$ and (Y|X) are generally entangled. This means that the unsupervised instances are helpful in learning the classifier and thus reduce the side effects of label noise. However, it remains elusive on how to exploit the causal information to handle the label-noise problem. We propose to model and make use of the causal process in order to correct the label-noise effect. Empirically, the proposed method outperforms all state-of-the-art methods on both synthetic and real-world label-noise datasets.


Note that data augmentation is used in this implementation and has achieved stronger results than the performance proposed in the original paper. The original implementation can be found at: https://github.com/a5507203/Instance-dependent-Label-noise-Learning-under-a-Structural-Causal-Model/tree/main

