import torch
from torch import nn
from model.sim_framework import CosineSimilarity,SimcseLearningModule
from model.model_baseline_finetune import UniModelFinetune


class Contrastive_frame(nn.Module):
    def __init__(self, args, device, model):
        super(Contrastive_frame, self).__init__()
        self.encoder = model
        self.device = device
        self.temp = args.temp
        self.alpha = args.alpha

    def forward(self, video_feature, video_mask, text_input_ids, text_mask, target):
        _, emb1, _, _ = self.encoder(video_feature, video_mask, text_input_ids, text_mask,target)
        _, emb2, _, _ = self.encoder(video_feature, video_mask, text_input_ids, text_mask,target)
        loss = 0 
        emb_avg1 = torch.nn.functional.normalize(emb1.to(self.device), dim=-1)
        emb_avg2 = torch.nn.functional.normalize(emb2.to(self.device), dim=-1)

        sim = emb_avg1 @ emb_avg2.t() / self.temp
        sim_targets = torch.zeros(sim.size()).to(self.device)
        sim_targets.fill_diagonal_(1)
        loss += -torch.sum(torch.nn.functional.log_softmax(sim, dim=1) * sim_targets, dim=1).mean() * self.alpha

        return emb1, _, _, loss

