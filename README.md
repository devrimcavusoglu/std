# Spatial-Channel Token Distillation for Vision MLPs
<a href="https://paperswithcode.com/paper/spatial-channel-token-distillation-for-vision"><img src="https://img.shields.io/badge/STD-temp?style=square&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJMYXllcl8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCIgdmlld0JveD0iMCAwIDUxMiA1MTIiIHN0eWxlPSJlbmFibGUtYmFja2dyb3VuZDpuZXcgMCAwIDUxMiA1MTI7IiB4bWw6c3BhY2U9InByZXNlcnZlIj4gPHN0eWxlIHR5cGU9InRleHQvY3NzIj4gLnN0MHtmaWxsOiMyMUYwRjM7fSA8L3N0eWxlPiA8cGF0aCBjbGFzcz0ic3QwIiBkPSJNODgsMTI4aDQ4djI1Nkg4OFYxMjh6IE0yMzIsMTI4aDQ4djI1NmgtNDhWMTI4eiBNMTYwLDE0NGg0OHYyMjRoLTQ4VjE0NHogTTMwNCwxNDRoNDh2MjI0aC00OFYxNDR6IE0zNzYsMTI4IGg0OHYyNTZoLTQ4VjEyOHoiLz4gPHBhdGggY2xhc3M9InN0MCIgZD0iTTEwNCwxMDRWNTZIMTZ2NDAwaDg4di00OEg2NFYxMDRIMTA0eiBNNDA4LDU2djQ4aDQwdjMwNGgtNDB2NDhoODhWNTZINDA4eiIvPjwvc3ZnPg%3D%3D&label=paperswithcode&labelColor=%23555&color=%2321b3b6&link=https%3A%2F%2Fpaperswithcode.com%2Fpaper%2Fspatial-channel-token-distillation-for-vision" alt="STD Implementation"></a>

![STD Framework](assets/std_framework.png)

A PyTorch implementation of the paper [Spatial-Channel Token Distillation for Vision MLPs](https://proceedings.mlr.press/v162/li22c.html). This project codebase is mostly based on the codebase of [DeiT from Facebook Research](https://github.com/facebookresearch/deit) and built on top of it with according changes, additions or removals.

This project and the repository is an outcome of collective paper implementation work conducted under [METU-CENG502](https://github.com/CENG502-Projects/CENG502-Spring2023). Refer to [STD implementation on CENG502](https://github.com/CENG502-Projects/CENG502-Spring2023/tree/main/Cavusoglu) to see the full and detailed report.

## Installation

For installation create the environment by executing the following cmd in the project root

```shell
conda env create -f environment.yml
```

## Training

Start training with 

```shell
python -m std.main --batch-size 128 --input-size 32 --patch-size 4 --model std-mlp-mixer --depth 8 --data-set CIFAR --data-path path/to/data --output_dir path/to/checkpoint_dir --teacher-model resnet32 resnet56
```

This will instantiate the run with last layer distillation only, to enable intermediate distillation, pass `--distill-intermediate`. The additional arguments can also be accessed, to see all arguments use the following command,

```shell
python -m std.main --help
```

## Evaluation

To evaluate a model, use the following command on appropriate model and arguments

```shell
python -m std.main --eval --resume path/to/model_folder --distillation-type none --teacher-model resnet32 resnet50 --model std-mlp-mixer --patch-size 4 --input-size 32 --data-set CIFAR --data-path path/to/dataset
```

This should give the following output for model STD-56

```
* Acc@1 76.850 Acc@5 94.170 loss 0.871
Accuracy of the network on the 10000 test images: 76.9
```

one important thing to notice here, if the model is trained with multiple-teacher setting, then you must pass `--teacher-model` argument accordingly to supply correct teacher count (all multi-teacher settings in the experiments were 2 teacher setting). Alternatively, instead of the model names you can pass anything (i.e. `--teacher-model 1 2` would work). Since this is evaluation only, instantiation of the teacher models do not take place, but this will inform the STD model to instantiate with correct layers and tokens, so that the model can be loaded correctly.

## Notes

Notes for implementation.

### Notes regarding MINE Regularization implementation:

The setup regarding the MINE regularization is not explicitly mentioned in the paper. There are mainly four parts that we set on our assumptions in the implementation:

- **Learning Rate:** The learning rate for the updates at Algorithm 1 (see 2.1.2.) is not mentioned. With some small experiments, we set and fixed the learning rate for MINE updates **for both statistics network and the vision model** as 0.01. We set this lr (as a bit high) due to the assumption of the sample size for the regularization. 
- **Sample size:** There is no explicit information in the paper regarding on how many samples this regularization has been done. We set this as a tunable argument in our implementation and set the default value as the batch size of the vision model. Thus, there are N samples used in each epoch for regularization where N is the original batch size used to train the vision model. Since the update uses a single batch and obviously **# of all batches >> 1**, for this regularization to have an effect we set the learning rate for regularization accordingly (a bit high compared to learning rate of the vision model).
  - **Note:** As this can be tuned to have higher, currently there is no data loader for this, and hence for higher sample size and with available memory limits, the training could potentially fail. Data loader for this part may come.
- **Statistics Network:** In the paper regarding the architecture of the statistics network, the information is given as 3 layer MLP with 512 dims. We assumed GELU as the activation function for all layers (same as in the MLP-Mixer layers), and assumed that there are no additional operations applied to the layers.
- **Selecting Samples:** For selecting joint samples (paired tokens) and marginal samples (unpaired tokens), we implemented a very naive way of derangement. The way follows the idea of shifting sample indices to 1 index right (i.e. indicies p=0,1,2,3,4 are used for both $T_S$ and $T_C$, and for unpaired tokens ($\overline{T_C}$) the indices becomes u=1,2,3,4,0.). This naive way is simple and guarantees derangement, but more complicated algorithms may be found. Moreover, this is safe to use as we collect the samples for regularization during training and effectively select a single instance from each batch and batches are randomly shuffled in each epoch.



## Contribution

To check if codestyle pass use

```shell
python -m scripts.run_code_style check
```

To reformat the codebase use

```shell
python -m scripts.run_code_style format
```

For easier setup, you can alternatively use the conda command

```shell
conda develop <path>
```

where `<path>` is the project root folder (not the source folder).

## License

This work contains the implementation of the methodology and study presented in the 
_Spatial-Channel Token Distillation for Vision MLPs_ paper. Also as the building block of the codebase, [facebookresearch/deit](https://github.com/facebookresearch/deit) is used, modified and adapted accordingly when necessary. The work here is licensed under MIT License (extending deit repository Apache license).
