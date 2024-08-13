#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import argparse
import json
import time
from pprint import PrettyPrinter

import numpy as np
import ruamel.yaml as yaml
import scipy

# import wandb
import torch
import torch.distributed as dist
from data_handling.datamodule import AudioCaptionDataModule
from data_handling.pretrain_dataset import pretrain_dataloader
from loguru import logger
from models.ase_model import ASE
from tools.losses import custom_contrastive_loss
from tools.optim_utils import cosine_lr, get_optimizer
from tools.utils import (
    AverageMeter,
    a2t,
    a2t_unique,
    compute_metrics,
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    is_main_process,
    log_results,
    set_logger,
    setup_seed,
    t2a,
    t2a_unique,
)
from tqdm import tqdm


def train(model, dataloader, optimizer, scheduler, device, epoch, use_new_loss=True):
    model.train()

    epoch_loss = AverageMeter()
    start_time = time.time()

    if is_dist_avail_and_initialized():
        dataloader.sampler.set_epoch(epoch)

    for batch_id, (audio, text, idx, extra_sentences_list) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        optimizer.zero_grad()

        step = len(dataloader) * (epoch - 1) + batch_id
        scheduler(step)
        # wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=step)

        audio = audio.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        loss = model(audio, text, idx, extra_sentences_list, use_new_loss=use_new_loss)

        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.cpu().item())

    elapsed_time = time.time() - start_time

    # wandb.log({"loss": epoch_loss.avg,
    #            "epoch": epoch})
    print(f"loss: {epoch_loss.avg}, epoch: {epoch}")

    return {"loss": epoch_loss.avg, "time": elapsed_time}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="settings/pretrain.yaml",
        type=str,
        help="Setting files",
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
    parser.add_argument("-m", "--model", default="Cnn14", type=str, help="Model name.")
    parser.add_argument("-a", "--max_length", default=30, type=int, help="Max length.")
    parser.add_argument("-s", "--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument(
        "-b",
        "--blacklist",
        # default="blacklist_exclude_ub8k_esc50_vggsound.json",
        type=str,
        help="Blacklist file.",
    )
    parser.add_argument(
        "--use_new_loss", default="True", type=str, help="If to use the new loss"
    )
    args = parser.parse_args()

    exp_name = args.exp_name

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # config["audio_encoder_args"]["type"] = args.model_type
    # config["audio_encoder_args"]["model"] = args.model
    # config["audio_args"]["max_length"] = args.max_length
    # config["optim_args"]["lr"] = args.lr
    # if args.blacklist:
    #     config["blacklist"] += args.blacklist
    # config["data_args"]["batch_size"] = args.batch_size

    # setup distribution mode
    init_distributed_mode(config["dist_args"])
    device = torch.device(config["device"])

    # setup seed
    seed = config["seed"] + get_rank()
    setup_seed(seed)

    exp_name = exp_name + f"_lr_{args.lr}_seed_{seed}"

    # wandb.init(
    #     project="AT-retrieval",
    #     name=exp_name,
    #     config=config
    # )

    # create pretrain dataloader
    dataloader = pretrain_dataloader(
        config,
        bucket=True,
        bucket_boundaries=(5, 30, 6),
        is_distributed=is_dist_avail_and_initialized(),
        num_tasks=get_world_size(),
        global_rank=get_rank(),
    )

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
        cp = torch.load(config.checkpoint, map_location="cpu")
        state_dict = cp["model"]

        optimizer.load_state_dict(cp["optimizer"])
        start_epoch = cp["epoch"] + 1
        model.load_state_dict(state_dict)

    # setup logger
    model_output_dir, log_output_dir = set_logger(exp_name)

    main_logger = logger.bind(indent=1)

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

    # load evaluation datamodule
    ac_datamodule = AudioCaptionDataModule(config, "AudioCaps")
    clotho_datamodule = AudioCaptionDataModule(config, "Clotho")
    syncaps_datamodule = AudioCaptionDataModule(config, "SynCaps")

    ac_val_loader = ac_datamodule.val_dataloader()
    clotho_val_loader = clotho_datamodule.val_dataloader()
    syncaps_val_loader = syncaps_datamodule.val_dataloader()

    loss_stats = []
    ac_recall_stats = []
    clotho_recall_stats = []
    syncaps_recall_stats = []

    for epoch in range(start_epoch, max_epoch + 1):
        main_logger.info(f"Training for epoch [{epoch}]")
        if args.use_new_loss == "False":
            train_statics = train(
                model,
                dataloader,
                optimizer,
                scheduler,
                device,
                epoch,
                use_new_loss=False,
            )
        else:
            train_statics = train(
                model,
                dataloader,
                optimizer,
                scheduler,
                device,
                epoch,
                use_new_loss=True,
            )
        loss = train_statics["loss"]
        elapsed_time = train_statics["time"]
        loss_stats.append(loss)

        main_logger.info(
            f"Training statistics:\tloss for epoch [{epoch}]: {loss:.3f},"
            f'\ttime: {elapsed_time:.1f}, lr: {optimizer.param_groups[0]["lr"]:.6f}.'
        )

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

        # validate on SynCaps and AudioCaps
        syncaps_metrics = validate_one(model, syncaps_val_loader, device)
        log_results(syncaps_metrics, "SynCaps", main_logger, test=False)
        syncaps_recall_stats.append(
            syncaps_metrics["t2a"][0] + syncaps_metrics["a2t"][0]
        )
        if syncaps_recall_stats[-1] >= max(syncaps_recall_stats) and is_main_process():
            sav_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
            }
            torch.save(sav_obj, str(model_output_dir) + "/syncaps_best_model.pt")

        # validate on AC and Clotho
        ac_metrics = validate(model, ac_val_loader, device)
        log_results(ac_metrics, "AudioCaps", main_logger, test=False)
        ac_recall_stats.append(ac_metrics["t2a"][0] + ac_metrics["a2t"][0])
        if ac_recall_stats[-1] >= max(ac_recall_stats) and is_main_process():
            sav_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
            }
            torch.save(sav_obj, str(model_output_dir) + "/ac_best_model.pt")

        # clotho_metrics = validate(model, clotho_val_loader, device)
        # log_results(clotho_metrics, "Clotho", main_logger, test=False)
        # clotho_recall_stats.append(clotho_metrics["t2a"][0] + clotho_metrics["a2t"][0])
        # if clotho_recall_stats[-1] >= max(clotho_recall_stats) and is_main_process():
        #     sav_obj = {
        #         "model": model_without_ddp.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "config": config,
        #         "epoch": epoch,
        #     }
        #     torch.save(sav_obj, str(model_output_dir) + "/clotho_best_model.pt")

    main_logger.info("Evaluation start...")
    ac_test_loader = ac_datamodule.test_dataloader()
    clotho_test_loader = clotho_datamodule.test_dataloader()

    model.load_state_dict(torch.load(str(model_output_dir) + "/best_model.pt")["model"])
    main_logger.info(
        f"Evaluation model with smallest loss... epoch:{torch.load(str(model_output_dir) + '/best_model.pt')['epoch']}"
    )
    ac_metrics = validate(model, ac_test_loader, device)
    log_results(ac_metrics, "AudioCaps", main_logger, test=True)
    clotho_metrics = validate(model, clotho_test_loader, device)
    log_results(clotho_metrics, "Clotho", main_logger, test=True)

    model.load_state_dict(
        torch.load(str(model_output_dir) + "/ac_best_model.pt")["model"]
    )
    main_logger.info(
        f"Evaluation best AudioCaps model... epoch:{torch.load(str(model_output_dir) + '/ac_best_model.pt')['epoch']}"
    )
    ac_metrics = validate(model, ac_test_loader, device)
    log_results(ac_metrics, "AudioCaps", main_logger, test=True)
    clotho_metrics = validate(model, clotho_test_loader, device)
    log_results(clotho_metrics, "Clotho", main_logger, test=True)

    model.load_state_dict(
        torch.load(str(model_output_dir) + "/clotho_best_model.pt")["model"]
    )
    main_logger.info(
        f"Evaluation best Clotho model... epoch:{torch.load(str(model_output_dir) + '/clotho_best_model.pt')['epoch']}"
    )
    ac_metrics = validate(model, ac_test_loader, device)
    log_results(ac_metrics, "AudioCaps", main_logger, test=True)
    clotho_metrics = validate(model, clotho_test_loader, device)
    log_results(clotho_metrics, "Clotho", main_logger, test=True)

    main_logger.info("Done.")
    # wandb.finish()


@torch.no_grad()
def validate_one(model, dataloader, device, lambda_new_loss=0):
    model.eval()
    audio_embeds_all, text_embeds_all = [], []
    total_loss = 0
    for batch_idx, (audio, text, idx, extra_sentences_list) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        audio = audio.to(device)

        audio_embeds = model.encode_audio(audio)
        text_embeds = model.encode_text(text)

        # if extra_sentences_list is not None:
        if lambda_new_loss != 0 and extra_sentences_list is not None:
            # Generate the mask: 1 if the sublist contains any non-empty string, 0 otherwise
            mask = torch.tensor(
                [
                    [
                        any(sentence != "" for sentence in sublist)
                        for sublist in extra_sentences_list
                    ]
                ]
            )
            extra_text_embeds = [
                model.encode_text(extra_text).T for extra_text in extra_sentences_list
            ]
            batch_extra_text_embeds = torch.stack(extra_text_embeds, dim=0)
            filtered_text_embeds = text_embeds[mask.squeeze(0)]
            filtered_batch_extra_text_embeds = batch_extra_text_embeds[mask.squeeze(0)]

        idx = idx.view(-1, 1)
        pos_idx = torch.eq(idx, idx.t()).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        sim_targets = sim_targets.to(device)
        # print("sim_targets", sim_targets)

        sim_a2t = audio_embeds @ text_embeds.t() / model.temp
        sim_t2a = text_embeds @ audio_embeds.t() / model.temp

        # if extra_sentences_list is not None:
        if lambda_new_loss != 0 and extra_sentences_list is not None:
            orig_loss = model.atc_loss(sim_a2t, sim_t2a, sim_targets)
            if lambda_new_loss == 0 or mask.sum().item() == 0:
                # print("using orig loss")
                loss = orig_loss
                pass
            else:
                new_loss = custom_contrastive_loss(
                    filtered_text_embeds,
                    filtered_batch_extra_text_embeds.transpose(1, 2),
                )
                loss = orig_loss + lambda_new_loss * new_loss
        else:
            loss = model.atc_loss(sim_a2t, sim_t2a, sim_targets)
        if model.embed_reg:
            loss = (
                loss
                + torch.mean(torch.abs(audio_embeds))
                / torch.sqrt(torch.sum(audio_embeds**2))
                + torch.mean(torch.abs(text_embeds))
                / torch.sqrt(torch.sum(text_embeds**2))
            )
        total_loss += loss.detach().item()

        audio_embeds_all.append(audio_embeds.cpu())
        text_embeds_all.append(text_embeds.cpu())

    audio_embeds_all = torch.cat(audio_embeds_all, dim=0).numpy()
    text_embeds_all = torch.cat(text_embeds_all, dim=0).numpy()

    # with open(
    #     "/scratch/shared/nfs2/oncescu/coding/libs/pt/WavCaps/audio_embeds_bat_lambda10_seed20_rev_cl.npy",
    #     "wb",
    # ) as f:
    #     np.save(f, audio_embeds_all)
    # with open(
    #     "/scratch/shared/nfs2/oncescu/coding/libs/pt/WavCaps/text_embeds_bat_lambda10_seed20_rev_cl.npy",
    #     "wb",
    # ) as f:
    #     np.save(f, text_embeds_all)

    # evaluate text to audio retrieval
    # print(t2a(audio_embeds_all, text_embeds_all))
    # print(t2a_unique(audio_embeds_all, text_embeds_all))
    print(t2v_metrics(text_embeds_all, audio_embeds_all))
    r1, r5, r10, r50, medr, meanr, mAP10 = t2a_unique(audio_embeds_all, text_embeds_all)
    print(compute_metrics(np.dot(text_embeds_all, audio_embeds_all.T)))

    # evaluate audio to text retrieval
    r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a = a2t_unique(
        audio_embeds_all, text_embeds_all
    )

    return {
        # "t2a": [
        #     t2a_results["R1"],
        #     t2a_results["R5"],
        #     t2a_results["R10"],
        #     t2a_results["R50"],
        #     t2a_results["MedR"],
        #     t2a_results["MeanR"],
        #     t2a_results["geometric_mean_R1-R5-R10"],
        # ],
        "t2a": [r1, r5, r10, r50, medr, meanr, mAP10],
        "a2t": [r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a],
        "val_loss": total_loss,
    }


@torch.no_grad()
def validate(model, dataloader, device, lambda_new_loss=0):
    model.eval()
    audio_embeds_all, text_embeds_all = [], []
    for batch_idx, (audio, text, idx, extra_sentences_list) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        audio = audio.to(device)

        audio_embeds = model.encode_audio(audio)
        text_embeds = model.encode_text(text)

        audio_embeds_all.append(audio_embeds.cpu())
        text_embeds_all.append(text_embeds.cpu())
        # break

    audio_embeds_all = torch.cat(audio_embeds_all, dim=0).numpy()
    text_embeds_all = torch.cat(text_embeds_all, dim=0).numpy()
    # with open("audio_embeds.npy", "wb") as f:
    #     np.save(f, audio_embeds_all)
    # with open("text_embeds.npy", "wb") as f:
    #     np.save(f, text_embeds_all)

    # evaluate text to audio retrieval
    # r1, r5, r10, r50, medr, meanr, mAP10 = t2a_unique(audio_embeds_all, text_embeds_all)
    r1, r5, r10, r50, medr, meanr, mAP10 = t2a(audio_embeds_all, text_embeds_all)

    # evaluate audio to text retrieval
    # r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a = a2t_unique(
    #     audio_embeds_all, text_embeds_all
    # )
    r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a = a2t(
        audio_embeds_all, text_embeds_all
    )

    return {
        "t2a": [r1, r5, r10, r50, medr, meanr, mAP10],
        "a2t": [r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a],
    }


def cols2metrics(cols, num_queries):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(cols == 0)) / num_queries
    metrics["R5"] = 100 * float(np.sum(cols < 5)) / num_queries
    metrics["R10"] = 100 * float(np.sum(cols < 10)) / num_queries
    metrics["R50"] = 100 * float(np.sum(cols < 50)) / num_queries
    metrics["MedR"] = np.median(cols) + 1
    metrics["MeanR"] = np.mean(cols) + 1
    stats = [metrics[x] for x in ("R1", "R5", "R10")]
    metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    print(f"Results calculated for {num_queries} queries")
    return metrics


def t2v_metrics(text_embd_, vid_embd_):
    """Compute retrieval metrics from a similiarity matrix.

    Args:
        sims (th.Tensor): N x M matrix of similarities between embeddings, where
             x_{i,j} = <text_embd[i], vid_embed[j]>
        query_masks (th.Tensor): mask any missing queries from the dataset (two videos
             in MSRVTT only have 19, rather than 20 captions)

    Returns:
        (dict[str:float]): retrieval metrics
    """
    # sims = np.array(torch.matmul(torch.tensor(text_embd_), (torch.tensor(vid_embd_)).t()))
    sims = np.dot(text_embd_, vid_embd_.T)
    assert sims.ndim == 2, "expected a matrix"

    num_queries, num_vids = sims.shape
    # print("num_queries", num_queries)
    # print("num_vids", num_vids)
    dists = -sims
    sorted_dists = np.sort(dists, axis=1)

    # The indices are computed such that they slice out the ground truth distances
    # from the psuedo-rectangular dist matrix
    queries_per_video = num_queries // num_vids
    gt_idx = [
        [
            np.ravel_multi_index([ii, jj], (num_queries, num_vids))
            for ii in range(jj * queries_per_video, (jj + 1) * queries_per_video)
        ]
        for jj in range(num_vids)
    ]
    gt_idx = np.array(gt_idx)
    # print("gt_idx shape", gt_idx.shape)
    # print(gt_idx)
    gt_dists = dists.reshape(-1)[gt_idx.reshape(-1)]
    gt_dists = gt_dists[:, np.newaxis]
    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT
    print("rows shape", rows.shape)
    # print("rows", rows)
    print("cols shape", cols.shape)
    # print("cols", cols)

    # --------------------------------
    # NOTE: Breaking ties
    # --------------------------------
    # We sometimes need to break ties (in general, these should occur extremely rarely,
    # but there are pathological cases when they can distort the scores, such as when
    # the similarity matrix is all zeros). Previous implementations (e.g. the t2i
    # evaluation function used
    # here: https://github.com/niluthpol/multimodal_vtt/blob/master/evaluation.py and
    # here: https://github.com/linxd5/VSE_Pytorch/blob/master/evaluation.py#L87) generally
    # break ties "optimistically".  However, if the similarity matrix is constant this
    # can evaluate to a perfect ranking. A principled option is to average over all
    # possible partial orderings implied by the ties. See # this paper for a discussion:
    #    McSherry, Frank, and Marc Najork,
    #    "Computing information retrieval performance measures efficiently in the presence
    #    of tied scores." European conference on information retrieval. Springer, Berlin,
    #    Heidelberg, 2008.
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.145.8892&rep=rep1&type=pdf

    break_ties = "optimistically"
    # break_ties = "averaging"

    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            _, idx = np.unique(rows, return_index=True)
            cols = cols[idx]
        elif break_ties == "averaging":
            # fast implementation, based on this code:
            # https://stackoverflow.com/a/49239335
            locs = np.argwhere((sorted_dists - gt_dists) == 0)

            # Find the split indices
            steps = np.diff(locs[:, 0])
            splits = np.nonzero(steps)[0] + 1
            splits = np.insert(splits, 0, 0)

            # Compute the result columns
            summed_cols = np.add.reduceat(locs[:, 1], splits)
            counts = np.diff(np.append(splits, locs.shape[0]))
            avg_cols = summed_cols / counts
            if False:
                print("Running slower code to verify rank averaging across ties")
                # slow, but more interpretable version, used for testing
                avg_cols_slow = [
                    np.mean(cols[rows == idx]) for idx in range(num_queries)
                ]
                assert np.array_equal(
                    avg_cols, avg_cols_slow
                ), "slow vs fast difference"
                print("passed num check")
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    if cols.size != num_queries:
        import ipdb

        ipdb.set_trace()
    assert cols.size == num_queries, msg

    return cols2metrics(cols, num_queries)


if __name__ == "__main__":
    main()
