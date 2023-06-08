# STD
Unoffical PyTorch implementation of the paper [Spatial-Channel Token Distillation for Vision MLPs](https://proceedings.mlr.press/v162/li22c.html).

## Installation

For installation create the environment by executing the following cmd in the project root

```shell
conda env create -f environment.yml
```

## Training

Start training with 

```shell
python -m std.main --batch-size 64 --data-set CIFAR --data-path None --output_dir /home/devrim/lab/gh/std/outputs --epochs 1 --mixup 0 --smoothing 0
```

## TODO
Taks to-do in the roadmap:

### Phase 1
- [X] MLP Mixer with STD
- [X] Implement MINE Regularization
- [ ] Refactor the training params to match with the paper & refactor params from transformer models to allMLP models
- [ ] Train CIFAR-100

### Phase 2
- [ ] CycleMLP with STD
- [ ] Multi-teacher implementation
- [ ] Confidence reweighting term for multi-teacher setting
- [ ] Last/Intermediate layer distillation
- [ ] Train ImageNet-1k
- [ ] Compare results with the paper

## Notes

Notes for implementation.

### Notes regarding MINE Regularization implementation:

- The number of samples that MINE algorithm uses is not specified in the paper. By default, it's equal to the batch size, but an argument added in the `main.py` as `n-mine-samples` to be specified if one wish to use different sample size other than the batch size for MINE. It can be set as 0 to not apply MINE regularization on the STD tokens. 
- It is not explicitly mentioned in the paper when the MINE regularization is applied on the weights, but we assumed that it is applied after casual weight updates (training).

### Notes regarding experimental setup

- In the paper, it's stated that "_the distillation tokens are inserted 2/3 positions_". It is assumed in the implementation that they refer to every 2nd of 3rd **network block** and not a single layer (e.g. Mixer block for MLPMixer).


## Contribution

To check if codestyle pass use

```shell
python -m scripts.run_code_style check
```

To reformat the codebase use

```shell
python -m scripts.run_code_style format
```
