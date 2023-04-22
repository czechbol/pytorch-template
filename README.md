# PyTorch Template Project
<p align="center">
  <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/torch?style=flat-square">
  <a href="https://github.com/czechbol/pytorch-template/graphs/commit-activity"><img src="https://img.shields.io/github/last-commit/czechbol/pytorch-template?style=flat-square" alt="Maintenance" /></a>
  <a href="https://github.com/czechbol/pytorch-template/actions"><img src="https://img.shields.io/github/actions/workflow/status/czechbol/pytorch-template/pre-commit.yml?style=flat-square" /></a>
  <a href="https://github.com/czechbol/pytorch-template/blob/master/LICENSE"><img src="https://img.shields.io/github/license/czechbol/pytorch-template?style=flat-square" alt="GPLv3 license" /></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square" alt="Formatted with Black" /></a>
</p>

PyTorch deep learning project made easy.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [PyTorch Template Project](#pytorch-template-project)
  - [Requirements](#requirements)
  - [Features](#features)
  - [Folder Structure](#folder-structure)
  - [Usage](#usage)
    - [Config file format](#config-file-format)
    - [Using config files](#using-config-files)
    - [Resuming from checkpoints](#resuming-from-checkpoints)
    - [Using Multiple GPU](#using-multiple-gpu)
  - [Customization](#customization)
    - [Project initialization](#project-initialization)
    - [Overriding config options](#overriding-config-options)
    - [Data Loader](#data-loader)
    - [Trainer](#trainer)
    - [Model](#model)
    - [Loss](#loss)
    - [Metrics](#metrics)
    - [Additional logging](#additional-logging)
    - [Testing](#testing)
    - [Validation data](#validation-data)
    - [Checkpoints](#checkpoints)
    - [Tensorboard Visualization](#tensorboard-visualization)
  - [Contribution](#contribution)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.8
* PyTorch >= 1.2 (2.0 recommended)
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))

## Features
* Clear folder structure which is suitable for many deep learning projects.
* `.json` config file support for convenient parameter tuning.
* Customizable command line options for more convenient parameter tuning.
* Checkpoint saving and resuming.
* Abstract base classes for faster development:
  * `BaseTrainer` handles checkpoint saving/resuming, training process logging, and more.
  * `BaseDataLoader` handles batch generation, data shuffling, and validation data splitting.
  * `BaseModel` provides basic model summary.

## Folder Structure
  ```
  pytorch-template/
  ├── LICENSE                   - Copyright information
  ├── README.md                 - This file
  ├── mypy.ini                  - Mypy configuration
  ├── new_project.py            - Initialize new project with template files
  ├── parse_config.py           - Class to handle config file and cli options
  ├── requirements-dev.txt      - Development requirements
  ├── requirements.txt          - Requirements to execute the project
  ├── test.py                   - Main script to start training
  ├── train.py                  - Main script to evaluate the model
  │
  ├── base/                     - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── configs                   - Configuration settings directory
  │   └── mnist.json            - Default MNIST model configuration
  │
  ├── data/                     - default directory for storing input data
  │
  ├── data_loader/              - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── logger/                   - module for tensorboard visualization and logging
  │   ├── logger_config.json
  │   ├── logger.py
  │   └── visualization.py
  │
  ├── model/                    - models, losses, and metrics
  │   ├── loss.py
  │   ├── metric.py
  │   └── model.py
  │
  ├── saved/
  │   ├── log/                  - default logdir for tensorboard and logging output
  │   └── models/               - trained models are saved here
  │
  ├── trainer/                  - trainers
  │   └── trainer.py
  │
  └── utils/                    - small utility functions
      └── util.py
  ```

## Usage
The code in this repo is an MNIST example of the template.
Try `python train.py -c configs/mnist.json` to run code.

### Config file format
Config files are in `.json` format:
```javascript
{
  "name": "Mnist_LeNet",        // training session name
  "n_gpu": 1,                   // number of GPUs to use for training.
  "arch": {
    "args": {},
    "type": "MnistModel"        // name of model architecture to train
  },
  "data_loader": {
    "type": "MnistDataLoader",  // selecting data loader
    "args": {
      "batch_size": 128,        // dataset path
      "data_dir": "data/",      // batch size
      "num_workers": 2,         // shuffle training data before splitting
      "shuffle": true,          // size of validation dataset. float(portion) or int(number of samples)
      "validation_split": 0.1   // number of cpu processes to be used for data loading
    }
  },
  "loss": "nll_loss",           // loss function
  "lr_scheduler": {
    "type": "StepLR",           // learning rate scheduler
    "args": {
      "gamma": 0.1,
      "step_size": 50
    }
  },
  "metrics": [                  // list of metrics to evaluate
    "accuracy",
    "top_k_acc"
  ],
  "optimizer": {
    "type": "Adam",
    "args": {
      "amsgrad": true,
      "lr": 0.001,              // learning rate
      "weight_decay": 0         // (optional) weight decay
    }
  },
  "trainer": {
    "early_stop": 10,           // number of epochs to wait before early stop. set 0 to disable.
    "epochs": 100,              // number of training epochs
    "monitor": "min val_loss",  // mode and metric for model performance monitoring. set 'off' to disable.
    "save_dir": "saved/",       // checkpoints are saved in save_dir/models/name
    "save_period": 1,           // save checkpoints every save_period epochs
    "tensorboard": true,        // enable tensorboard visualization
    "verbosity": 2              // 0: quiet, 1: per epoch, 2: full
  }
}

```

Add addional configurations if you need.

### Using config files
Copy the default `configs/mnist.json` config file, modify it, then run:

  ```
  python train.py -c configs/your_config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py -r saved/models/path/to/checkpoint
  ```

### Using Multiple GPU
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py -d 2,3 -c config.json
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```

## Customization

### Project initialization
Use the `new_project.py` script to make your new project directory with template files.
`python new_project.py ../NewProject` then a new project folder named 'NewProject' will be made.
This script will filter out unneccessary files like cache, git files or readme file.

### Overriding config options

Changing values of config file is a clean, safe and easy way of tuning hyperparameters. However, sometimes
it is better to have command line options if some values need to be changed too often or quickly.

This template uses the configurations stored in the json file by default, we provide some arguments to override said configuration.

The default values disable this behavior.

  ```
    -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        Optionally set learning rate. Replaces the value from the given config file (default: -1)
    -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Optionally set the batch size. Replaces the value from the given config file (default: -1)
  ```


### Data Loader
* **Writing your own data loader**

1. **Inherit ```BaseDataLoader```**

    `BaseDataLoader` is a subclass of `torch.utils.data.DataLoader`, you can use either of them.

    `BaseDataLoader` handles:
    * Generating next batch
    * Data shuffling
    * Generating validation data loader by calling
    `BaseDataLoader.split_validation()`

* **DataLoader Usage**

  `BaseDataLoader` is an iterator, to iterate through batches:
  ```python
  for batch_idx, (x_batch, y_batch) in data_loader:
      pass
  ```
* **Example**

  Please refer to `data_loader/data_loaders.py` for an MNIST data loading example.

### Trainer
* **Writing your own trainer**

1. **Inherit ```BaseTrainer```**

    `BaseTrainer` handles:
    * Training process logging
    * Checkpoint saving
    * Checkpoint resuming
    * Reconfigurable performance monitoring for saving current best model, and early stop training.
      * If config `monitor` is set to `max val_accuracy`, which means then the trainer will save a checkpoint `model_best.pth` when `validation accuracy` of epoch replaces current `maximum`.
      * If config `early_stop` is set, training will be automatically terminated when model performance does not improve for given number of epochs. This feature can be turned off by passing 0 to the `early_stop` option, or just deleting the line of config.

2. **Implementing abstract methods**

    You need to implement `_train_epoch()` for your training process, if you need validation then you can implement `_valid_epoch()` as in `trainer/trainer.py`

* **Example**

  Please refer to `trainer/trainer.py` for MNIST training.

* **Iteration-based training**

  `Trainer.__init__` takes an optional argument, `len_epoch` which controls number of batches(steps) in each epoch.

### Model
* **Writing your own model**

1. **Inherit `BaseModel`**

    `BaseModel` handles:
    * Inherited from `torch.nn.Module`
    * `__str__`: Modify native `print` function to prints the number of trainable parameters.

2. **Implementing abstract methods**

    Implement the foward pass method `forward()`

* **Example**

  Please refer to `model/model.py` for a LeNet example.

### Loss
Custom loss functions can be implemented in 'model/loss.py'. Use them by changing the name given in "loss" in config file, to corresponding name.

### Metrics
Metric functions are located in 'model/metric.py'.

You can monitor multiple metrics by providing a list in the configuration file, e.g.:
  ```json
  "metrics": ["accuracy", "top_k_acc"],
  ```

### Additional logging
If you have additional information to be logged, in `_train_epoch()` of your trainer class, merge them with `log` as shown below before returning:

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log.update(additional_log)
  return log
  ```

### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

### Validation data
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, then it will return a data loader for validation of size specified in your config file.
The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).

**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`

### Checkpoints
You can specify the name of the training session in config files:
  ```json
  "name": "MNIST_LeNet",
  ```

The checkpoints will be saved in `save_dir/name/timestamp/checkpoint_epoch_n`, with timestamp in mmdd_HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }
  ```

### Tensorboard Visualization
This template supports Tensorboard visualization by using either  `torch.utils.tensorboard` or [TensorboardX](https://github.com/lanpa/tensorboardX).

1. **Install**

    If you are using pytorch 1.1 or higher, install tensorboard by 'pip install tensorboard>=1.14.0'.

    Otherwise, you should install tensorboardx. Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).

2. **Run training**

    Make sure that `tensorboard` option in the config file is turned on.

    ```
     "tensorboard" : true
    ```

3. **Open Tensorboard server**

    Type `tensorboard --logdir saved/log/` at the project root, then server will open at `http://localhost:6006`

By default, values of loss and metrics specified in config file, input images, and histogram of model parameters will be logged.
If you need more visualizations, use `add_scalar('tag', data)`, `add_image('tag', image)`, etc in the `trainer._train_epoch` method.
`add_something()` methods in this template are basically wrappers for those of `tensorboardX.SummaryWriter` and `torch.utils.tensorboard.SummaryWriter` modules.

**Note**: You don't have to specify current steps, since `WriterTensorboard` class defined at `logger/visualization.py` will track current steps.

## Contribution
Check out [CONTRIBUTING.md](./CONTRIBUTING.md)

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements
This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95).\
This project is a fork of [victoresque/pytorch-template](https://github.com/victoresque/pytorch-template)
