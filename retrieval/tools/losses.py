#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import util


class AudioTextContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sim_a2t, sim_t2a, sim_targets=None):
        if sim_targets is None:
            sim_targets = torch.zeros(sim_a2t.size()).to(sim_a2t.device)
            sim_targets.fill_diagonal_(1)

        loss_a2t = -torch.sum(F.log_softmax(sim_a2t, dim=1) * sim_targets, dim=1).mean()

        loss_t2a = -torch.sum(F.log_softmax(sim_t2a, dim=1) * sim_targets, dim=1).mean()

        loss_atc = (loss_a2t + loss_t2a) / 2
        return loss_atc


class NTXent(nn.Module):
    def __init__(self, temperature=0.07):
        super(NTXent, self).__init__()
        self.loss = nn.LogSoftmax(dim=1)
        self.tau = temperature

    def forward(self, audio_embeds, text_embeds, labels):
        n = audio_embeds.shape[0]

        a2t = util.cos_sim(audio_embeds, text_embeds) / self.tau
        t2a = util.cos_sim(text_embeds, audio_embeds) / self.tau

        # mask = labels.expand(n, n).eq(labels.expand(n, n).t()).to(a2t.device)
        # mask_diag = mask.diag()
        # mask_diag = torch.diag_embed(mask_diag)
        # mask = mask ^ mask_diag
        #
        # a2t_loss = - self.loss(a2t).masked_fill(mask, 0).diag().mean()
        # t2a_loss = - self.loss(t2a).masked_fill(mask, 0).diag().mean()

        t2a_loss = -self.loss(t2a).mean()
        a2t_loss = -self.loss(a2t).mean()

        loss = 0.5 * a2t_loss + 0.5 * t2a_loss

        return loss


def custom_contrastive_loss(queries, concatenated_embeddings, alpha=0.2):
    """
    queries: Tensor of shape [N, D] containing N query embeddings.
    concatenated_embeddings: Tensor of shape [N, 4, D] containing embeddings of 2 positive examples followed by 2 negative examples for each query.
    alpha: Margin by which positives should be closer to the query than negatives.
    """
    # Split the concatenated embeddings into positives and negatives
    positives, negatives = (
        concatenated_embeddings[:, :2, :],
        concatenated_embeddings[:, 2:, :],
    )

    # Calculate similarities
    pos_sim = torch.einsum("nd,nkd->nk", [queries, positives])
    neg_sim = torch.einsum("nd,nkd->nk", [queries, negatives])

    # Compute loss
    loss = 0
    for i in range(2):  # Assuming 2 positive and 2 negative examples
        loss += F.relu(alpha - pos_sim[:, i] + neg_sim[:, i]).mean()

    return loss
