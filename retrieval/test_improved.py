#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import argparse
import os
import time
from pprint import PrettyPrinter

# import wandb
import numpy as np
import ruamel.yaml as yaml
import torch
import torch.distributed as dist
from data_handling.datamodule import AudioCaptionDataModule
from data_handling.pretrain_dataset import pretrain_dataloader
from loguru import logger
from models.ase_model import ASE
from pretrain import validate, validate_one
from tools.optim_utils import cosine_lr, get_optimizer
from tools.utils import (
    AverageMeter,
    a2t,
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    is_main_process,
    log_results,
    set_logger,
    setup_seed,
    t2a,
)
from tqdm import tqdm

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="settings/train.yaml", type=str, help="Setting files"
    )
    parser.add_argument(
        "-n",
        "--exp_name",
        default="exp_name",
        help="name of this experiment.",
    )
    parser.add_argument("-l", "--lr", default=5e-5, type=float, help="Learning rate.")
    parser.add_argument(
        "-t", "--model_type", default="cnn", type=str, help="Model type."
    )
    parser.add_argument("-m", "--model", default="htsat", type=str, help="Model name.")
    parser.add_argument("-a", "--max_length", default=30, type=int, help="Max length.")
    parser.add_argument(
        "-d", "--dataset", default="AudioCaps", type=str, help="Dataset."
    )
    parser.add_argument(
        "--lambda_new_loss",
        default=0,
        type=float,
        help="If not 0, then new loss is used times lambda_new_loss.",
    )
    parser.add_argument(
        "--seed",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--train_filename",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--val_filename",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--test_filename",
        default=None,
        type=str,
    )

    args = parser.parse_args()

    exp_name = args.exp_name

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # config["audio_encoder_args"]["type"] = args.model_type
    # config["audio_encoder_args"]["model"] = args.model
    config["seed"] = args.seed

    config["data_args"]["train"] = (
        args.train_filename
        if args.train_filename
        else config["data_args"]["train"]
        if "train" in config["data_args"]
        else "train"
    )
    config["data_args"]["val"] = (
        args.val_filename
        if args.val_filename
        else config["data_args"]["val"]
        if "val" in config["data_args"]
        else "val"
    )
    config["data_args"]["test"] = (
        args.test_filename
        if args.test_filename
        else config["data_args"]["test"]
        if "test" in config["data_args"]
        else "test"
    )

    # setup distribution mode
    init_distributed_mode(config["dist_args"])
    device = torch.device(config["device"])

    # setup seed
    seed = config["seed"] + get_rank()
    setup_seed(seed)
    torch.use_deterministic_algorithms(True)

    config["optim_args"]["lr"] = args.lr
    # config["data_args"]["dataset"] = args.dataset
    args_dataset = config["data_args"]["dataset"]
    exp_name = (
        exp_name
        + f"_{args_dataset}_lr_{args.lr}_seed_{seed}"
        + "_lambda_"
        + str(args.lambda_new_loss)
        + "_"
        + config["data_args"]["train"]
        + "_"
        + config["data_args"]["val"]
    )

    # wandb.init(project="AT-retrieval", name=exp_name, config=config)

    # load evaluation datamodule

    datamodule = AudioCaptionDataModule(config, args_dataset)

    # setup model
    model = ASE(config)
    model = model.to(device)
    # wandb.watch(model)

    cp = torch.load(config["pretrain_path"], map_location=device)
    state_dict = cp["model"]
    print(f'loading pretrained model {config["pretrain_path"]}')
    model.load_state_dict(state_dict)
    logger.info(f"Loaded pretrain model from {config['pretrain_path']}")

    # setup logger
    model_output_dir, log_output_dir = set_logger(exp_name)

    main_logger = logger.bind(indent=1)
    logger.info(f'using seed {config["seed"]}')

    # print training settings
    printer = PrettyPrinter()
    main_logger.info("Training setting:\n" f"{printer.pformat(config)}")

    main_logger.info(
        f"Total numer of parameters: {sum([i.numel() for i in model.parameters()])}"
    )

    # if is_dist_avail_and_initialized():
    #     model = torch.nn.parallel.DistributedDataParallel(model)

    validate_fct = {1: validate_one, 5: validate}

    main_logger.info("Evaluation start...")
    names_test_sets = {
        "SynCaps": ["test", "test_reverse", "test_replaced"],
        "AudioCaps": [
            # "test_audioset_1_rearranged",
            # "test_audioset_1_rearranged_future_past",
            # "test_audioset_1_rearranged_future_past_reverse",
            # "test_audioset_1_rearranged_future_past_replace",
            # "test_audioset_1",
            # "test_audioset_1_future_past",
            # "test_audioset_1_future_past_reverse",
            # "test_audioset_1_future_past_replace",
            "test_full",
            ###############
            "test_audioset_5_future_past",
            "test_audioset_5_future_past_reverse",
            "test_audioset_5_future_past_replace",
            "test_audioset_5_rearranged",
            "test_audioset_5_rearranged_future_past",
            "test_audioset_5_rearranged_future_past_reverse",
            "test_audioset_5_rearranged_future_past_replace",
            #############3
            # "test_audioset_1_rearranged_temporal_prep_future_past",
            # "test_audioset_1_rearranged_temporal_prep_future_past_reverse",
            # "test_audioset_1_rearranged_temporal_prep_future_past_replace",
            # "test_audioset_5_better_times",
            # "test_audioset_1_rearranged_temporal_prep",
            # "test_audioset_1_rearranged_temporal_prep_reverse",
            # "test_audioset_1_rearranged_temporal_prep_replace",
        ],
        "Clotho": [
            "test",
            "test_future_past_5captions",
            "test_future_past_reverse_5captions",
            "test_future_past_replace_5captions",
            # "bat_test",
            # "bat_test_reverse",
            # "bat_test_ac",
            # "bat_test_ac_rev",
            # "bat_test_clotho",
            # "bat_test_clotho_rev",
        ],
    }

    for name_test_set in names_test_sets[args_dataset]:
        setup_seed(seed)
        config["data_args"]["test"] = name_test_set
        datamodule = AudioCaptionDataModule(config, args_dataset)
        main_logger.info(f"Evaluation on {name_test_set}...")
        test_loader = datamodule.test_dataloader()

        main_logger.info(
            f"Validation start, using function {validate_fct[datamodule.test_set.num_captions_per_audio].__name__}"
        )

        metrics = validate_fct[datamodule.test_set.num_captions_per_audio](
            model, test_loader, device, lambda_new_loss=args.lambda_new_loss
        )
        log_results(metrics, config["data_args"]["dataset"], main_logger, test=True)

    main_logger.info("Done.")
    # wandb.finish()


if __name__ == "__main__":
    main()
