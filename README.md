# Coconet-pytorch

This is a standalone pytorch implementation of [Counterpoint by Convolution](https://arxiv.org/abs/1903.07227) (Coconet)
. This implementation is based on [kevindonoghue's implementation](https://github.com/kevindonoghue/coconet-pytorch),
while changed few parameters and network structures for reproducing the network in the paper.

The code in this repo is modified
from [kevindonoghue's implementation](https://github.com/kevindonoghue/coconet-pytorch)
by [Yusong Wu](https://github.com/lukewys) and [Kyle Kastner](https://github.com/kastnerkyle).

Although the code is modified to reproduce the original paper, the loss in here is just plain cross-entropy between
input and output. Because the use of cross-entropy, it filters out all the training data with rest notes. 

In the original paper (eq.3)
and [original implementation](https://github.com/magenta/magenta/blob/188bbf922aa36bc437ae45e99b2e5803074677dc/magenta/models/coconet/lib_graph.py#L190)
, the loss is not counted for back-propagation where input is not masked (because the objective would be simply to copy
input to the output), and the loss is scaled by 1/(T-num_unmasked+1). In this implementation we found the un-scaled loss
still produce decent output.

## Requirements

pytorch, pretty_midi, midi2audio, pyfluidsynth

## Generation

To generate random Bach chorales in a length of 8 bars, simply clone the repo and run `python generate.py`.

The pre-trained network is trained using batch size of 64 and 50000 steps of updates.

## Train

To train the network, simply download the "Jsb16thSeparated.npz"
in [here](https://github.com/czhuang/JSB-Chorales-dataset), and run the `train.py`.

