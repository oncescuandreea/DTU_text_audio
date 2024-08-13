#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

# import wandb
from loguru import logger
from sentence_transformers import util


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_logger(exp_name):
    log_output_dir = Path("outputs", exp_name, "logging")
    model_output_dir = Path("outputs", exp_name, "models")
    log_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stdout,
        format="{time: YYYY-MM-DD at HH:mm:ss} | {message}",
        level="INFO",
        filter=lambda record: record["extra"]["indent"] == 1,
    )
    logger.add(
        log_output_dir.joinpath("output.txt"),
        format="{time: YYYY-MM-DD at HH:mm:ss} | {message}",
        level="INFO",
        filter=lambda record: record["extra"]["indent"] == 1,
    )

    return model_output_dir, log_output_dir


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    # Adjust the seed for each worker
    setup_seed(42 + worker_id)


# def setup_for_distributed(is_master):
#     """
#        This function disables printing when not in master process
#        """
#     import builtins as __builtin__

#     builtin_print = __builtin__.print

#     def print(*args, **kwargs):
#         force = kwargs.pop("force", False)
#         if is_master or force:
#             builtin_print(*args, **kwargs)

#     __builtin__.print = print


def setup_for_distributed(is_master):
    import builtins as __builtin__

    original_print = __builtin__.print

    def conditional_print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            original_print(*args, **kwargs)

    try:
        # Temporarily override print
        __builtin__.print = conditional_print

        # Your code where overridden print is needed
        # ...

    finally:
        # Make sure to restore the original print function
        __builtin__.print = original_print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    args["distributed"] = False
    return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args["rank"] = int(os.environ["RANK"])
        args["world_size"] = int(os.environ["WORLD_SIZE"])
        args["gpu"] = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args["rank"] = int(os.environ["SLURM_PROCID"])
        args["gpu"] = args["rank"] % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args["distributed"] = False
        return

    args["distributed"] = True

    torch.cuda.set_device(args["gpu"])
    args["dist_backend"] = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args["rank"], args["dist_url"]),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args["dist_backend"],
        init_method=args["dist_url"],
        world_size=args["world_size"],
        rank=args["rank"],
    )
    torch.distributed.barrier()
    setup_for_distributed(args["rank"] == 0)


def log_results(results, dataset, main_logger, test=False):
    if test:
        pre = "test"
    else:
        pre = "val"
    main_logger.info(
        "{}: Caption to audio: r1: {:.2f}, r5: {:.2f}, "
        "r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}, mAP10: {:.3f}".format(
            dataset, *results["t2a"]
        )
    )
    main_logger.info(
        "{}: Audio to caption: r1: {:.2f}, r5: {:.2f}, "
        "r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}, mAP10: {:.3f}".format(
            dataset, *results["a2t"]
        )
    )
    # wandb.log({
    #     f"{dataset}:{pre}_t2a/r1": results["t2a"][0],
    #     f"{dataset}:{pre}_t2a/r5": results["t2a"][1],
    #     f"{dataset}:{pre}_t2a/r10": results["t2a"][2],
    #     f"{dataset}:{pre}_t2a/mAP10": results["t2a"][-1],
    # })

    # wandb.log({
    #     f"{dataset}:{pre}_a2t/r1": results["a2t"][0],
    #     f"{dataset}:{pre}_a2t/r5": results["a2t"][1],
    #     f"{dataset}:{pre}_a2t/r10": results["a2t"][2],
    #     f"{dataset}:{pre}_a2t/mAP10": results["a2t"][-1],
    # })


def remove_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def a2t(audio_embs, cap_embs, return_ranks=False):
    # audio to caption retrieval
    num_audios = int(audio_embs.shape[0] / 5)

    ranks = np.zeros(num_audios)
    top1 = np.zeros(num_audios)
    AP10 = np.zeros(num_audios)
    for index in range(num_audios):
        # get query audio
        audio = audio_embs[5 * index]

        # compute scores
        # d = audio @ cap_embs.T
        d = util.cos_sim(torch.Tensor(audio), torch.Tensor(cap_embs)).squeeze(0).numpy()
        inds = np.argsort(d)[::-1]

        inds_map = []

        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
            if tmp < 10:
                inds_map.append(tmp + 1)
        inds_map = np.sort(np.array(inds_map))
        # calculate average precision
        if len(inds_map) != 0:
            AP10[index] = np.sum((np.arange(1, len(inds_map) + 1) / inds_map)) / 5
        else:
            AP10[index] = 0.0
        ranks[index] = rank
        top1[index] = inds[0]
    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    mAP10 = 100.0 * np.sum(AP10) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, mAP10, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr, mAP10


def t2a(audio_embs, cap_embs, return_ranks=False):
    # caption to audio retrieval
    print("audio_embeds shape:", audio_embs.shape, "cap_embeds shape:", cap_embs.shape)
    num_audios = int(audio_embs.shape[0] / 5)

    audios = np.array([audio_embs[i] for i in range(0, audio_embs.shape[0], 5)])

    ranks = np.zeros(5 * num_audios)
    top1 = np.zeros(5 * num_audios)

    for index in range(num_audios):
        # get query captions
        queries = cap_embs[5 * index : 5 * index + 5]

        # compute scores
        # d = queries @ audios.T
        d = util.cos_sim(torch.Tensor(queries), torch.Tensor(audios)).numpy()

        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    mAP10 = 100.0 * np.sum(1 / (ranks[np.where(ranks < 10)[0]] + 1)) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, mAP10, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr, mAP10


def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    # ind = sx - d
    # ind = np.where(ind == 0)
    epsilon = 1e-6  # Small threshold for floating-point comparison
    ind = np.where(np.abs(sx - d) <= epsilon)
    ind = ind[1]
    metrics = {}
    metrics["R1"] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics["R5"] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics["R10"] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics["MR"] = np.median(ind) + 1
    metrics["MedianR"] = metrics["MR"]
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return (
        metrics["R1"],
        metrics["R5"],
        metrics["R10"],
        0,
        metrics["MedianR"],
        metrics["MeanR"],
        0,
    )
    # return metrics


def t2a_unique(audio_embs, cap_embs, return_ranks=False):
    # Assuming audio_embs.shape[0] == cap_embs.shape[0]
    num_items = audio_embs.shape[
        0
    ]  # Number of items now directly corresponds to the number of embeddings

    ranks = np.zeros(num_items)
    top1 = np.zeros(num_items)

    for index in range(num_items):
        # Get a single query caption for each audio
        query = cap_embs[index].reshape(
            1, -1
        )  # Reshape to ensure it's 2D for matrix operations

        # Compute scores
        d = util.cos_sim(torch.Tensor(query), torch.Tensor(audio_embs)).numpy()
        # print("d", d)

        inds = np.argsort(d[0])[::-1]  # Sort the scores for this query
        # print("inds", inds)
        ranks[index] = np.where(inds == index)[0][0]
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    mAP10 = 100.0 * np.sum(1 / (ranks[np.where(ranks < 10)[0]] + 1)) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, mAP10, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr, mAP10


def a2t_unique(audio_embs, cap_embs, return_ranks=False):
    # Assuming audio_embs.shape[0] == cap_embs.shape[0]
    num_audios = audio_embs.shape[0]  # Use the actual number of audio embeddings

    ranks = np.zeros(num_audios)
    top1 = np.zeros(num_audios)
    AP10 = np.zeros(num_audios)

    for index in range(num_audios):
        # Get query audio
        audio = audio_embs[index].reshape(
            1, -1
        )  # Ensure audio is 2D for matrix operations

        # Compute scores
        d = util.cos_sim(torch.Tensor(audio), torch.Tensor(cap_embs)).squeeze(0).numpy()
        inds = np.argsort(d)[::-1]

        # Find the rank and top1
        rank = np.where(inds == index)[0][0]
        ranks[index] = rank
        top1[index] = inds[0]

        # Calculate average precision for top 10 (AP@10), simplified since we're looking at a 1:1 mapping
        AP10[index] = 1.0 / (rank + 1) if rank < 10 else 0.0

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    mAP10 = 100.0 * np.sum(AP10) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, mAP10, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr, mAP10
