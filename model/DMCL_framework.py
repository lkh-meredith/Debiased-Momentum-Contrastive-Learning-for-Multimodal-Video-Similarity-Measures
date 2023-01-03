import torch
from torch import nn
from model.sim_framework import CosineSimilarity,SimcseLearningModule
from model.model_baseline_finetune import UniModelFinetune


class Momentum_contrastive_frame(nn.Module):
    def __init__(self,args,device, pretrained_dict, model):
        super(Momentum_contrastive_frame, self).__init__()
        self.encoder = model
        self.task = []

        if args.tag is True:
            self.task = ['tag']
        self.encoder_m = UniModelFinetune(args, task=self.task)
        finetune_model_m_dict = self.encoder_m.state_dict()

        finetune_model_m_dict.update(pretrained_dict)
 
        self.encoder_m.load_state_dict(finetune_model_m_dict, strict=False)
        self.encoder_m.to(device)
        self.device = device
    
        self.alpha = args.alpha
        self.beta = args.beta
        self.momentum_debias = args.momentum_debias

        self.temp = args.temp
        self.momentum = args.momentum
        self.queue_size = args.queue_size

        self.model_pairs=[[self.encoder,self.encoder_m]]

        self.copy_params()

        # create the queue
        self.sim = args.sim
        if self.momentum_debias is True:
            if self.sim == 'cos_mean':
                self.register_buffer("representation_queue", torch.randn(self.queue_size, 256))
                self.register_buffer("representation_cls_queue", torch.randn(self.queue_size, 256))
                self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
                self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
                self.representation_queue = nn.functional.normalize(self.representation_queue)
                self.representation_cls_queue = nn.functional.normalize(self.representation_cls_queue)

        else:
            if self.sim == 'cos_mean':
                self.register_buffer("representation_queue", torch.randn(self.queue_size, 256))
                self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
                self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
                self.representation_queue = nn.functional.normalize(self.representation_queue)


    def forward(self, video_feature, video_mask, text_input_ids, text_mask, target):
        _, emb, emb_cls, _ = self.encoder(video_feature, video_mask, text_input_ids, text_mask,target=target)
        loss = 0
        emb_avg = torch.nn.functional.normalize(emb.to(self.device), dim=-1)
        emb_cls = torch.nn.functional.normalize(emb_cls, dim=-1)

        with torch.no_grad():
            self._momentum_update()
            _, emb_m, emb_cls_m, _  = self.encoder_m(video_feature, video_mask, text_input_ids, text_mask)  # emb: batch*len*dim, mask:batch*len
            emb_m_avg = torch.nn.functional.normalize(emb_m.to(self.device), dim=-1)
            if self.momentum_debias is True:
                emb_all = torch.cat([emb_m_avg, self.representation_queue.clone().detach().to(self.device)], dim=0)
                emb_m_cls = torch.nn.functional.normalize(emb_cls_m, dim=-1)
                emb_all_cls = torch.cat([emb_m_cls, self.representation_cls_queue.clone().detach().to(self.device)], dim=0)
                soft_cls_sim = emb_cls @ emb_all_cls.t() / self.temp

                sim_targets = torch.zeros(soft_cls_sim.size()).to(self.device)
                sim_targets.fill_diagonal_(1)

            else:
                emb_all = torch.cat([emb_m_avg, self.representation_queue.clone().detach().to(self.device)], dim=0)

        if self.momentum_debias:
            sim = emb_avg @ emb_all.t() / self.temp
            sim_targets = self.beta * torch.nn.functional.softmax(soft_cls_sim, dim=1) + (1 - self.beta) * sim_targets
            loss += -torch.sum(torch.nn.functional.log_softmax(sim, dim=1) * sim_targets, dim=1).mean() * self.alpha
            self._dequeue_and_enqueue(emb_m_avg, emb_m_cls)
        else:
            sim = emb_avg @ emb_all.t() / self.temp
            sim_targets = torch.zeros(sim.size()).to(self.device)
            sim_targets.fill_diagonal_(1)
            loss += -torch.sum(torch.nn.functional.log_softmax(sim, dim=1) * sim_targets, dim=1).mean() * self.alpha
            self._dequeue_and_enqueue(emb_m_avg,None)

        return emb, _, _, loss

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, representation_feat,representation_cls):
        batch_size = representation_feat.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        if self.momentum_debias is True:
            if self.sim == 'cos_mean':
                self.representation_queue[ptr:ptr + batch_size,:] = representation_feat
                self.representation_cls_queue[ptr:ptr + batch_size, :] = representation_cls
        else:
            if self.sim == 'cos_mean':
                self.representation_queue[ptr:ptr + batch_size, :] = representation_feat

        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr
