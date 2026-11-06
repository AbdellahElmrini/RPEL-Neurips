This is a brief guide on how to use this repository to reproduce the results from our paper :"Robust and Scalable Collaborative Learning via Pull-Based Epidemic Communication"


## Installation 

To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```


## Reproducing the experimenst

To reproduce the CIFAR-10 experiments :
- run the script run.py as follows (the parameter supercharge launches multiple experiments on parallel, use it with precaution to avoid memory overflow)

``python3 run.py --supercharge 2``

To reproduce the MNIST experiments :
- run the script mnist_run.py in a similar way.

``python3 mnist_run.py --supercharge 2``

To reproduce the fixed-graph baselines: 
- run the scripts fx_mnist_run.py or fx_cifar10_run.py



To reproduce the simulations (Figure 3 in the paper)

- run the simulations.py script

``python3 simulations.py``


To test with different parameters, please edit the parameters inside the run.py and mnist_run.py scripts. 

Look into src/train_p2p.py, or src/fx_train_p2p.py files to see a description of the arguments



