# Coconet-pytorch

This is a standalone pytorch implementation of [Counterpoint by Convolution](https://arxiv.org/abs/1903.07227) (Coconet)
. This implementation is based on [kevindonoghue's implementation](https://github.com/kevindonoghue/coconet-pytorch),
while changed few parameters and network structures for reproducing the network in the paper.

The code in this repo is modified
from [kevindonoghue's implementation](https://github.com/kevindonoghue/coconet-pytorch)
by [Yusong Wu](https://github.com/lukewys) and [Kyle Kastner](https://github.com/kastnerkyle).

## Requirements

pytorch, pretty_midi, midi2audio, pyfluidsynth

## Generation

To generate random Bach chorales in a length of 8 bars, simply clone the repo and run `python generate.py`.

The pre-trained network is trained using batch size of 64 and 50000 steps of updates.

## Train

To train the network, simply download the "Jsb16thSeparated.npz"
in [here](https://github.com/czhuang/JSB-Chorales-dataset), and run the `train.py`.

