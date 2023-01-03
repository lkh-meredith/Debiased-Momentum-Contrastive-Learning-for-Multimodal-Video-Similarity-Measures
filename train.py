import os, math, random, time, sys, gc, sys, json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import collections
import logging
from imp import reload
reload(logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(f"train_{time.strftime('%m%d_%H%M', time.localtime())}.log"),
        logging.StreamHandler()
    ]
)
import argparse
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from dataset.record_trans import record_transform
from dataset.video_text_dataset import VideoTextDataset
from model.model_baseline import UniModel
from model.model_baseline_finetune import UniModelFinetune
from utils.utils import set_random_seed
from transformers import AdamW, get_cosine_schedule_with_warmup
from model.DMCL_framework import Debiased_Momentum_contrastive_frame
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.cuda.current_device()
torch.cuda._initialized = True

gc.enable()

os.environ['NUMEXPR_MAX_THREADS'] = '8'
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_pred_and_loss(args, model, item, framework=None,mode=None):
    video_feature1 = item['frame_features'].to(device)
    input_ids1 = item['id'].to(device)
    attention_mask1 = item['mask'].to(device)
    video_mask1 = item['frame_mask'].to(device)
    task = None

    sim, loss = 0, 0

    video_feature2 = item['frame_features2'].to(device)
    input_ids2 = item['id2'].to(device)
    attention_mask2 = item['mask2'].to(device)
    video_mask2 = item['frame_mask2'].to(device)
    label = item['label'].to(device)
    sim_output = None
    if mode == 'eval':
        _, emb1, emb1_cls, _ = model(video_feature1, video_mask1, input_ids1, attention_mask1, target=None)
        _, emb2, emb2_cls, _ = model(video_feature2, video_mask2, input_ids2, attention_mask2, target=None)
        compute_sim = nn.CosineSimilarity(dim=-1)
        if args.sim == 'cos_mean':
            sim = compute_sim(emb1, emb2)
        if args.sim == 'cos_cls':
            sim = compute_sim(emb1_cls, emb2_cls)

        loss = nn.MSELoss()(sim.view(-1), label.view(-1))

    if mode == 'sup':
        target1 = None
        if args.tag:
 
            target1 = item['target'].to(device)
        target2 = None
        if args.tag:
            target2 = item['target2'].to(device)
        _, emb1, emb1_cls, loss1 = model(video_feature1, video_mask1, input_ids1, attention_mask1, target=target1)
        _, emb2, emb2_cls, loss2 = model(video_feature2, video_mask2, input_ids2, attention_mask2, target=target2)
       
        compute_sim = nn.CosineSimilarity(dim=-1)
        sim_mean = compute_sim(emb1, emb2)
        kl_loss = 0
        if args.momentum_debias:
            sim_cls = compute_sim(emb1_cls, emb2_cls)
            kl_loss = F.kl_div(sim_cls.softmax(dim=-1).log(), sim_mean.softmax(dim=-1), reduction='batchmean')

        loss = nn.MSELoss()(sim_mean.view(-1), label.view(-1)) + (loss1 + loss2) * 0.05 + kl_loss # + nn.MSELoss()(sim.view(-1), sim_cls.view(-1))  #

    if mode == 'unsup':
        target1 = None
        target =None
        if args.tag:
            target1 = item['target'].to(device)
        target2 = None
        if args.tag:
            target2 = item['target2'].to(device)
            target = torch.cat([target1, target2], dim=0)
        video_feature = torch.cat([video_feature1, video_feature2], dim=0)
        video_mask = torch.cat([video_mask1, video_mask2], dim=0)
        input_ids = torch.cat([input_ids1, input_ids2], dim=0)
        attention_mask = torch.cat([attention_mask1, attention_mask2], dim=0)

        if framework is not None:
            _, _, _, loss = framework(video_feature, video_mask, input_ids, attention_mask, target=target)
    return sim, loss, label


def eval(args, model, data_loader):
    model.eval()
    sim_l, label_l, loss_l = [], [], []
    with torch.no_grad():
        for batch_num, item in enumerate(data_loader):
            sim, loss, label = get_pred_and_loss(args, model, item, mode='eval')
            if loss is not None:
                loss_l.append(loss.to("cpu"))
            if sim is not None:
                sim_l.append(sim.to("cpu").numpy())
            if label is not None:
                label_l.append(label.to("cpu").numpy())

    if len(sim_l) != 0:
        sim_l = np.concatenate(sim_l)
    if len(label_l) != 0:
        label_l = np.concatenate(label_l)

    return sim_l, label_l, np.mean(loss_l)


def train(args, model, train_loader, val_loader, optimizer, scheduler=None, framework=None,save_path=None):
    step = 0
    best_val_loss = None
    best_spearman = None
    start = time.time()
    for epoch in range(args.epochs):
        for _, item in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            _, loss1, _ = get_pred_and_loss(args, model, item, mode='sup')  # sup
            _, loss2, _ = get_pred_and_loss(args, model, item, framework=framework, mode='unsup')
            loss = loss1 + loss2

            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            logging.info(f"Epoch={epoch + 1}/{args.epochs}|step={step:3}|train_loss={loss:6.5}")
            if step == 20 or (step % 500 == 0 and step > 0):
                elapsed_seconds = time.time() - start  # Evaluate the model on val_loader.
                pred, label, val_loss = eval(args, model, val_loader)
                val_spearman = scipy.stats.spearmanr(label, pred).correlation
                improve_str = ''

                if not best_val_loss or val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    best_spearman = val_spearman

                    torch.save(model.state_dict(), save_path)
                    improve_str = f"|New best_val_loss={best_val_loss:6.4}"

                logging.info(f"Epoch={epoch + 1}/{args.epochs}|step={step:3}|val_spearman={val_spearman:6.4}|time={elapsed_seconds:0.3}s" + improve_str)
                start = time.time()
            step += 1

        elapsed_seconds = time.time() - start  # Evaluate the model on val_loader.
        pred, label, val_loss = eval(args, model, val_loader)
        val_spearman = scipy.stats.spearmanr(label, pred).correlation
        improve_str = ''
        if not best_val_loss or val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_spearman = val_spearman
            torch.save(model.state_dict(), save_path)
            improve_str = f"|New best_val_loss={best_val_loss:6.4}"

        logging.info(f"Epoch={epoch + 1}/{args.epochs}|step={step:3}|val_spearman={val_spearman:6.4}|time={elapsed_seconds:0.3}s" + improve_str)
        start = time.time()

    return best_val_loss, best_spearman


#########################################
# Train pairwise model
#########################################
if __name__=="__main__":
    logging.info("Start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='./input/pretrain-model/hfl/chinese-bert-wwm-ext', type=str)
    parser.add_argument("--model_pretrained_path", default='./save_model/model_pretrain_bert.pth', type=str)
    parser.add_argument("--save_model_path", default='./save_model/finetune_bert', type=str)
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--learning_rate", default=5e-6, type=float)
    parser.add_argument("--unsup_constras", default=True, type=bool)
    parser.add_argument("--momentum_debias", default=True, type=bool)
    parser.add_argument("--tag", default = True, type=bool)
    parser.add_argument("--num_tag_class", default=64903, type=int)
    parser.add_argument("--alpha", default=0.05, type=float)
    parser.add_argument("--temp", default=0.2, type=float)
    parser.add_argument("--dropout_prob", default=0.05, type=float)
    parser.add_argument("--sim", default='cos_mean', type=str)  #cos_cls
    parser.add_argument("--queue_size", default=1024, type=int)
    parser.add_argument("--momentum", default=0.999, type=float)
    parser.add_argument("--beta", default=0.5, type=float)
    args = parser.parse_args()
    set_random_seed(args.seed)
    model_save = f'batch{args.batch_size}'

    if args.unsup_constras is True:
        logging.info(f"Config-type: unsup_constras")
        model_save = f"unsup_constras_{args.sim}_alpha{args.alpha}_queue{args.queue_size}_momen{args.momentum}_t{args.temp}_drop{args.dropout_prob}_" + model_save
    if args.momentum_debias is True:
        logging.info(f"Config-type: momentum_debias")
        model_save = f"beta{args.beta}_" + model_save
    if not args.unsup_constras and not args.momentum_debias:
        logging.info(f"Config-type: districtly compute {args.sim}")
        model_save = f"{args.sim}_" + model_save
    if args.tag is True:
        logging.info(f"Config-type: tag")
        model_save = f"tag_" + model_save

    logging.info(f"Saving model:" + f"{args.save_model_path}_" + model_save)

    fintune_model_save_path = f"{args.save_model_path}_epoch{args.epochs}_{model_save}.pth"
    pair_dataset_path = '../MVSE/annotations/pairwise.json'
    pair_video_feature_path = '../MVSE/features/clip/pairwise/'

    logging.info("Load data")
    trans = record_transform(model_path=args.model_path, tag_file='./tag_list.csv', get_title=True, get_tagid=True)
    pair_label = pd.read_csv('../MVSE/annotations/pairwise.tsv', sep='\t', header=None, dtype={0: str, 1: str})

    pair_dataset = VideoTextDataset(pair_dataset_path, pair_video_feature_path, record_transform=trans,
                                    label_pair=pair_label, pairwise=True, tag=args.tag)
    logging.info("Load pair dataset finish")

    val_dataset_path = '../MVSE/annotations/test-dev.json'
    val_video_feature_path = '../MVSE/features/clip/test-dev/dev/'
    val_pair_label = pd.read_csv('../MVSE/annotations/test_dev.tsv', sep='\t', header=None, dtype={0: str, 1: str})
    val_trans = record_transform(model_path=args.model_path, tag_file=None, get_title=True)
    val_dataset = VideoTextDataset(val_dataset_path, val_video_feature_path, record_transform=val_trans,
                                   label_pair=val_pair_label, pairwise=True, tag=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, num_workers=4)
    logging.info("Load eval dataset finish")

    pairwise_size = len(pair_dataset) // args.batch_size
    train_loader = DataLoader(pair_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=4,
                              pin_memory=True)
    logging.info(f"pairwise_size = {pairwise_size},eval_pairwise_size = {len(val_dataset) // args.batch_size}")

    total_steps = args.epochs * pairwise_size
    warmup_steps = pairwise_size
    logging.info(f'Total train steps={total_steps}, warmup steps={warmup_steps}')

    # model
    if args.tag is True:
        model = UniModel(args, task=['tag'])
        finetune_model = UniModelFinetune(args, task=['tag'])

    else:
        model = UniModel(args, task=[])
        finetune_model = UniModelFinetune(args, task=[])
    model.load_state_dict(torch.load(args.model_pretrained_path), strict=False)
    pretrained_dict = model.state_dict()

    finetune_model_dict = finetune_model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in finetune_model_dict}

    finetune_model_dict.update(pretrained_dict)
    finetune_model.load_state_dict(finetune_model_dict, strict=False)
    finetune_model.to(device)

    framework = None
    if args.unsup_constras is True:
        framework = Debiased_Momentum_contrastive_frame(args, device, pretrained_dict, finetune_model)
        framework.to(device)

    param_optimizer = list(finetune_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    # train
    val_loss, val_spearman = train(args, finetune_model, train_loader, val_loader, optimizer, scheduler=scheduler,
                                   framework=framework, save_path=fintune_model_save_path)
    logging.info(f"val MSE={val_loss}")
    logging.info(f"val Spearman={val_spearman}")

    finetune_model.load_state_dict(torch.load(fintune_model_save_path))  # load best model
    finetune_model.to(device)

    eval_dataset_path = '../MVSE/annotations/test-std.json'
    eval_video_feature_path = '../MVSE/features/clip/test-dev/std/'
    eval_pair_label = pd.read_csv('../MVSE/annotations/test_std.tsv', sep='\t', header=None, dtype={0: str, 1: str})
    eval_trans = record_transform(model_path=args.model_path, tag_file=None, get_title=True)
    eval_dataset = VideoTextDataset(eval_dataset_path, eval_video_feature_path, record_transform=eval_trans,
                                    label_pair=eval_pair_label, pairwise=True, tag=False)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, drop_last=False, num_workers=4)

    pred, label, eval_loss = eval(args, finetune_model, eval_loader)
    eval_spearman = scipy.stats.spearmanr(label, pred).correlation
    logging.info(f"eval_spearman:{eval_spearman}")
    logging.info(f"eval_mse:{eval_loss}")
    logging.info("Eval finish")

