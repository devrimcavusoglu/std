# STD
Unoffical PyTorch implementation of the paper [Spatial-Channel Token Distillation for Vision MLPs](https://proceedings.mlr.press/v162/li22c.html).


## Training

Start training with 

```shell
python -m std.main --batch-size 64 --data-set CIFAR --data-path None --output_dir /home/devrim/lab/gh/std/outputs --epochs 1 --mixup 0 --smoothing 0
```

## TODO
Taks to-do in the roadmap:

- [x] MLP Mixer with STD
- [ ] CycleMLP with STD
- [ ] Multi-teacher implementation
- [ ] Last/Intermediate layer distillation
- [ ] Implement MINE
- [ ] Train CIFAR-100
- [ ] Compare results with the paper

