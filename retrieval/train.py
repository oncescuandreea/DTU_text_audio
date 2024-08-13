#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import argparse
import json
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


def train(model, dataloader, optimizer, scheduler, device, epoch, lambda_new_loss=0):
    model.train()

    epoch_loss = AverageMeter()
    start_time = time.time()

    if is_dist_avail_and_initialized():
        dataloader.sampler.set_epoch(epoch)
    all_audios = []
    all_texts = []
    for batch_id, (audio, text, idx, extra_sentences_list) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        optimizer.zero_grad()

        step = len(dataloader) * (epoch - 1) + batch_id
        scheduler(step)
        # wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=step)

        audio = audio.to(device, non_blocking=True)
        # all_audios.append(audio)
        # all_texts.extend(text)
        idx = idx.to(device, non_blocking=True)

        loss = model(
            audio, text, idx, extra_sentences_list, lambda_new_loss=lambda_new_loss
        )

        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.cpu().item())

    elapsed_time = time.time() - start_time
    # all_audios = torch.cat(all_audios, dim=0)
    # with open(f"all_audios_epoch_{epoch}_again_shuffle_seed19.pt", "wb") as f:
    #     torch.save(all_audios, f)

    # with open(f"all_texts_epoch_{epoch}_again_shuffle_seed19.json", "w") as f:
    #     json.dump(all_texts, f)
    # wandb.log({"loss": epoch_loss.avg,
    #            "epoch": epoch})

    print({"loss": epoch_loss.avg, "epoch": epoch})

    return {"loss": epoch_loss.avg, "time": elapsed_time}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="settings/train.yaml", type=str, help="Setting files"
    )
    parser.add_argument(
        "-n",
        "--exp_name",
        default="exp_name",
        type=str,
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
    # config["audio_args"]["max_length"] = args.max_length
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
    config["seed"] = args.seed

    # setup distribution mode
    init_distributed_mode(config["dist_args"])
    device = torch.device(config["device"])

    # setup seed
    seed = config["seed"] + get_rank()
    print(f"using seed {seed}")
    setup_seed(seed)
    # torch.use_deterministic_algorithms(True)

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
    print(exp_name)

    # wandb.init(project="AT-retrieval", name=exp_name, config=config)

    # load evaluation datamodule

    datamodule = AudioCaptionDataModule(config, args_dataset)

    dataloader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # setup model
    model = ASE(config)
    model = model.to(device)
    # wandb.watch(model)

    # setup optim utils
    optimizer = get_optimizer(
        model.parameters(),
        lr=config["optim_args"]["lr"],
        betas=config["optim_args"]["betas"],
        eps=config["optim_args"]["eps"],
        momentum=config["optim_args"]["momentum"],
        optimizer_name=config["optim_args"]["optimizer_name"],
    )
    scheduler = cosine_lr(
        optimizer,
        base_lr=config["optim_args"]["lr"],
        warmup_length=config["optim_args"]["warmup_epochs"] * len(dataloader),
        steps=len(dataloader) * config["training"]["epochs"],
    )
    start_epoch = 1
    max_epoch = config["training"]["epochs"]

    if config["resume"]:
        cp = torch.load(config["checkpoint"], map_location="cpu")
        state_dict = cp["model"]

        optimizer.load_state_dict(cp["optimizer"])
        start_epoch = cp["epoch"] + 1
        print(f"starting from start_epoch {start_epoch}")
        model.load_state_dict(state_dict)
    elif config["pretrain"]:
        cp = torch.load(config["pretrain_path"], map_location=device)
        state_dict = cp["model"]
        print(f'loading pretrained model {config["pretrain_path"]}')
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pretrain model from {config['pretrain_path']}")

    # setup logger
    model_output_dir, log_output_dir = set_logger(exp_name)

    main_logger = logger.bind(indent=1)
    # main_logger.info(f'using seed {config["seed"]}')

    # print training settings
    printer = PrettyPrinter()
    main_logger.info("Training setting:\n" f"{printer.pformat(config)}")

    main_logger.info(
        f"Total numer of parameters: {sum([i.numel() for i in model.parameters()])}"
    )
    main_logger.info(
        f"Size of training set: {len(dataloader.dataset)}, size of batches: {len(dataloader)}"
    )

    model_without_ddp = model
    if is_dist_avail_and_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model)
        model_without_ddp = model.module

    loss_stats = []
    recall_stats = []
    loss_val_stats = []
    validate_fct = {1: validate_one, 5: validate}

    for epoch in range(start_epoch, max_epoch + 1):
        main_logger.info(f"Training for epoch [{epoch}]")

        train_statics = train(
            model,
            dataloader,
            optimizer,
            scheduler,
            device,
            epoch,
            lambda_new_loss=args.lambda_new_loss,
        )
        loss = train_statics["loss"]
        elapsed_time = train_statics["time"]
        loss_stats.append(loss)

        main_logger.info(
            f'Training statistics:\tloss for epoch [{epoch}]: {loss:.3f},'
            f'\ttime: {elapsed_time:.1f}, lr: {optimizer.param_groups[0]["lr"]:.6f}.'
        )

        if is_main_process():
            sav_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
            }

            torch.save(sav_obj, str(model_output_dir) + "/current_epoch_model.pt")

        if loss <= min(loss_stats) and is_main_process():
            sav_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
            }

            torch.save(sav_obj, str(model_output_dir) + "/best_model.pt")

        if is_dist_avail_and_initialized():
            dist.barrier()
            torch.cuda.empty_cache()

        # validate on AC and Clotho
        main_logger.info(
            f"Validation start, using function {validate_fct[datamodule.val_set.num_captions_per_audio].__name__}"
        )

        metrics = validate_fct[datamodule.val_set.num_captions_per_audio](
            model, val_loader, device, lambda_new_loss=args.lambda_new_loss
        )
        if datamodule.val_set.num_captions_per_audio == 1:
            loss_val_stats.append(metrics["val_loss"])
        log_results(metrics, config["data_args"]["dataset"], main_logger, test=False)
        recall_stats.append(metrics["t2a"][0] + metrics["a2t"][0])
        if recall_stats[-1] >= max(recall_stats) and is_main_process():
            sav_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
            }
            torch.save(
                sav_obj,
                str(model_output_dir)
                + f"/recall_best_model_{datamodule.val_set.num_captions_per_audio}.pt",
            )
        if datamodule.val_set.num_captions_per_audio == 1:
            if loss_val_stats[-1] <= min(loss_val_stats) and is_main_process():
                sav_obj = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch,
                }
                torch.save(
                    sav_obj,
                    str(model_output_dir)
                    + f"/loss_best_model_{datamodule.val_set.num_captions_per_audio}.pt",
                )

    main_logger.info("Evaluation start...")
    names_test_sets = {
        "SynCaps": ["test", "test_reverse", "test_replaced"],
        "AudioCaps": [
            "test_audioset_1_rearranged",
            "test_audioset_1_rearranged_future_past",
            "test_audioset_1_rearranged_future_past_reverse",
            "test_audioset_1_rearranged_future_past_replace",
            "test_audioset_1",
            "test_audioset_1_future_past",
            "test_audioset_1_future_past_reverse",
            "test_audioset_1_future_past_replace",
            "test_full",
            "test_audioset_5_future_past",
            "test_audioset_5_future_past_reverse",
            "test_audioset_5_future_past_replace",
            "test_audioset_5_rearranged",
            "test_audioset_5_rearranged_future_past",
            "test_audioset_5_rearranged_future_past_reverse",
            "test_audioset_5_rearranged_future_past_replace",
        ],
        "Clotho": [
            "test",
            "test_future_past_5captions",
            "test_future_past_reverse_5captions",
            "test_future_past_replace_5captions",
        ],
    }

    for name_test_set in names_test_sets[args_dataset]:
        config["data_args"]["test"] = name_test_set
        print(f"using seed {seed}")
        setup_seed(seed)
        datamodule = AudioCaptionDataModule(config, args_dataset)
        main_logger.info(f"Evaluation on {name_test_set}...")
        test_loader = datamodule.test_dataloader()
        main_logger.info(
            f"Validation start, using function {validate_fct[datamodule.test_set.num_captions_per_audio].__name__}"
        )

        # evaluate the model with the smallest loss on train set
        model.load_state_dict(
            torch.load(str(model_output_dir) + "/best_model.pt")["model"]
        )
        main_logger.info(
            f"Evaluation model with smallest loss on train set...epoch:{torch.load(str(model_output_dir) + '/best_model.pt')['epoch']}"
        )
        metrics = validate_fct[datamodule.test_set.num_captions_per_audio](
            model, test_loader, device, lambda_new_loss=args.lambda_new_loss
        )
        log_results(metrics, config["data_args"]["dataset"], main_logger, test=True)

        # evaluate the model with the highest recall on val set
        model.load_state_dict(
            torch.load(
                str(model_output_dir)
                + f"/recall_best_model_{datamodule.val_set.num_captions_per_audio}.pt"
            )["model"]
        )
        main_logger.info(
            f"Evaluation model with highest recall on validation...epoch:{torch.load(str(model_output_dir) + f'/recall_best_model_{datamodule.val_set.num_captions_per_audio}.pt')['epoch']}"
        )
        metrics = validate_fct[datamodule.test_set.num_captions_per_audio](
            model, test_loader, device, lambda_new_loss=args.lambda_new_loss
        )
        log_results(metrics, config["data_args"]["dataset"], main_logger, test=True)

        if datamodule.val_set.num_captions_per_audio == 1:
            # evaluate the model with the smallest loss on val set
            model.load_state_dict(
                torch.load(
                    str(model_output_dir)
                    + f"/loss_best_model_{datamodule.val_set.num_captions_per_audio}.pt"
                )["model"]
            )
            main_logger.info(
                f"Evaluation model with smallest loss on validation...epoch:{torch.load(str(model_output_dir) + f'/loss_best_model_{datamodule.val_set.num_captions_per_audio}.pt')['epoch']}"
            )
            metrics = validate_fct[datamodule.test_set.num_captions_per_audio](
                model, test_loader, device, lambda_new_loss=args.lambda_new_loss
            )
            log_results(metrics, config["data_args"]["dataset"], main_logger, test=True)

    main_logger.info("Done.")
    # wandb.finish()


if __name__ == "__main__":
    main()
