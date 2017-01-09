# neuroWorks

Neural network babysteps with tensorflow.


## NN - Architectures for EM - Segmentation

* Ciresan: http://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images
* U-Net: http://link.springer.com/chapter/10.1007/978-3-319-24574-4_28 / http://link.springer.com/chapter/10.1007/978-3-319-46723-8_49
* FusionNet: https://arxiv.org/abs/1612.05360
* ZNN: https://arxiv.org/abs/1508.04843


## Questions for Nasim:

* Normalization: How and where? What about BN-Layers?
* Randomization of training data: Is permuting after every cycle the right thing to do?


## Some ideas:

* Prunining for NNs start with fully connected NN and drop connections that become too weak.
Basically that is L0 (infeasible) / L1 loss <- NN with L1 papers? But could also drop connections once their absolute value becomes too small.
Probably with droping rate increasing during the learning process?
* End-to-end learning for connectomics


## Side projects:

* DeepQuoran
