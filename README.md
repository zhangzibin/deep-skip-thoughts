# deep-skip-thoughts

This is an Torch implementation of [Skip-Thought Vectors](http://arxiv.org/abs/1506.06726). The original implementation is [here](https://github.com/ryankiros/skip-thoughts). You may read the paper for more details.

Most of the code is borrowed from [Andrej Karpathy's Char-Rnn](https://github.com/karpathy/char-rnn). Thanks for Andrej.

My implementation is different from Ryan Kiros's at some point such as the optimizer but the main architecture is the same. Mutil-layer is supported for better result.

I'm still working on this project. There may be some problems in my code and it's result haven't be tested by now. But you can try this bleeding-edge version if you wish.


## Requirements

This code is written in Lua and requires [Torch](http://torch.ch/).
Additionally, you need to install the `nngraph` and `optim` packages using [LuaRocks](https://luarocks.org/) which you will be able to do after installing Torch:

```bash
$ luarocks install nngraph 
$ luarocks install optim
```

If you'd like to use CUDA GPU computing, you'll first need to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), then the `cutorch` and `cunn` packages:

```bash
$ luarocks install cutorch
$ luarocks install cunn
```


## Usage

The following command will tell you everything to do.

```bash
$ th rnn2rnn.lua --help
```
