
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
# sys.path.append("..")
from model.mask_lm import MaskLM,MaskVideo
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertModel
from transformers import BertModel


class UniModelFinetune(nn.Module):
    def __init__(self, args,task=[]):
        super().__init__()
        model_path = args.model_path
        uni_bert_cfg = BertConfig.from_pretrained(f'{model_path}/config.json')

        self.newfc_hidden = torch.nn.Linear(uni_bert_cfg.hidden_size, 256)
        self.proj_head = nn.Sequential(
            nn.ReLU(),
            torch.nn.Linear(256, 256)
        )

        if args.dropout_prob is not None:
            uni_bert_cfg.hidden_dropout_prob = args.dropout_prob

        self.task = task
        if 'tag' in self.task: 
            self.newfc_tag = torch.nn.Linear(uni_bert_cfg.hidden_size , args.num_tag_class)  #uni_bert_cfg.hidden_size, 1024->10000


        self.roberta = UniBertForMaskedLM.from_pretrained(model_path, config=uni_bert_cfg)


    def forward(self, video_feature, video_mask, text_input_ids, text_mask, target=None):
        loss, pred = 0, None

        video_feature_all, video_mask_all, text_input_ids_all, text_mask_all = video_feature, video_mask, text_input_ids, text_mask
        features = self.roberta(video_feature_all, video_mask_all, text_input_ids_all, text_mask_all)

        embedding_mean = self.newfc_hidden(torch.mean(features[:, 1:, :], 1))  # batch*1*256
        # embedding_mean = torch.mean(embedding[:,1:,:], 1) # batch*1*768
        embedding_cls = self.proj_head(self.newfc_hidden(features[:, 0, :]))  # batch*1*256
        # embedding_cls = self.newfc_hidden(features[:, 0, :])  # batch*1*256

        if 'tag' in self.task:
            pred = self.newfc_tag(torch.relu(features[:, 0, :]))
            if target is not None:
                tagloss = nn.BCEWithLogitsLoss(reduction="mean")(pred.view(-1), target.view(-1))
                loss += tagloss* 1250

        return pred, embedding_mean, embedding_cls, loss


class UniBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = UniBert(config)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask):
        encoder_outputs= self.bert(video_feature, video_mask, text_input_ids, text_mask)

        return encoder_outputs

# Copied from 2021_QQ_AIAC_Tack1_1st/blob/main/job1/qqmodel/qq_uni_model.py
class UniBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.video_fc = torch.nn.Linear(512, config.hidden_size)
        self.video_embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask):
        text_emb = self.embeddings(input_ids=text_input_ids)

        # text input is [CLS][SEP] text [SEP]
        cls_emb = text_emb[:, 0:1, :]  
        text_emb = text_emb[:, 1:, :]  

        cls_mask = text_mask[:, 0:1]  
        text_mask = text_mask[:, 1:]

        # reduce frame feature dimensions : 512 -> 768
        video_feature = self.video_fc(video_feature)
        video_emb = self.video_embeddings(inputs_embeds=video_feature)

        # [CLS] Video [SEP] Text [SEP]
        embedding_output = torch.cat([cls_emb, video_emb, text_emb], 1)  

        mask = torch.cat([cls_mask, video_mask, text_mask], 1)
        mask1 = mask[:, None, None, :]
        mask1 = (1.0 - mask1) * -10000.0

        encoder_outputs = self.encoder(embedding_output, attention_mask=mask1)['last_hidden_state']  

        return encoder_outputs
