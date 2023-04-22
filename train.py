import argparse
from typing import Callable, List, Type, Union

import numpy as np
import torch
from base.base_model import BaseModel

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config: ConfigParser) -> None:
    logger = config.get_logger("train")

    # setup data_loader instances
    data_loader: Callable = config.init_obj("data_loader", module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model: Union[BaseModel, torch.nn.DataParallel]
    model = config.init_obj("arch", module_arch)
    print(type(model))
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion: Callable = getattr(module_loss, config["loss"])
    metrics: List[Callable] = [getattr(module_metric, met) for met in config["metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer: Type[torch.optim.Optimizer] = config.init_obj(
        "optimizer", torch.optim, trainable_params
    )
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="",
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default="",
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default="",
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-l",
        "--learning_rate",
        default=-1,
        type=float,
        help="Optionally set learning rate. Replaces the value from the given config file (default: -1)",
    )
    args.add_argument(
        "-b",
        "--batch_size",
        default=-1,
        type=int,
        help="Optionally set the batch size. Replaces the value from the given config file (default: -1)",
    )
    parsed_args = args.parse_args()

    config = ConfigParser.from_args(parsed_args)
    main(config)
