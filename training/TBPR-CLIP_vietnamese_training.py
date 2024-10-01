import os, sys
print(os.getcwd())
from pathlib import Path
ROOT_PATH = Path('../../paper_clones/TBPS-CLIP').resolve()
sys.path.append(str(ROOT_PATH))
IMAGE_PATH = Path('../../DATASET').resolve()
sys.path.append(str(IMAGE_PATH))

import os, json
from torchinfo import summary
import random
import wandb
import time, datetime
from pathlib import Path
import torch
from misc.build import load_checkpoint, cosine_scheduler, build_optimizer
from misc.data import build_pedes_data
from misc.eval import test
from misc.utils import parse_config, init_distributed_mode, set_seed, is_master, is_using_distributed, AverageMeter
from model.tbps_model import clip_vitb, CLIP
from options import get_args
import nltk; nltk.download('stopwords')

anno_path = IMAGE_PATH/'CUHK-PEDES/reid_translate.json'
# objs = json.load(open(anno_path))

config_path = ROOT_PATH/'config/config.yaml'
config = parse_config(config_path)

set_seed(config)
config['log']['print_period'] = 1
config['model']['checkpoint'] = ROOT_PATH/'checkpoint/ViT-B-16.pt'
config['anno_dir'] = anno_path.parent/'vietnamese'
# config['anno_dir'] = ROOT_PATH/'annotation/CUHK-PEDES'

config['image_dir'] = IMAGE_PATH/'CUHK-PEDES/imgs'
config['device'] = 'cuda'
config['model']['use_gather'] = False
config['data']['batch_size'] = 50
config['model']['saved_path'] = ROOT_PATH/"checkpoint"
config['experiment']['text_length'] = 400
config['model']['embed_dim'] = 512
config['schedule']['epoch_warmup'] = 2
config['schedule']['weight_decay'] = 0.02
config['schedule']['epoch'] = 100
config['experiment']['nitc_ratio'] = 1.0
config['experiment']['citc_ratio'] = 0.15
meters = {
    "loss": AverageMeter(),
    "nitc_loss": AverageMeter(),
    "ss_loss": AverageMeter(),
    "citc_loss": AverageMeter(),
    "ritc_loss": AverageMeter(),
    "mlm_loss": AverageMeter(),
    "id_loss": AverageMeter(),
}
best_rank_1 = 0.0
best_epoch = 0

dataloader = build_pedes_data(config)
train_loader = dataloader['train_loader']
test_loader = dataloader['test_loader']
num_classes = len(train_loader.dataset.person2text)

from sentence_transformers import SentenceTransformer, util
import transformers
from transformers import CLIPProcessor, CLIPModel
clip_processor = transformers.CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
tokenizer = transformers.AutoTokenizer.from_pretrained('sentence-transformers/clip-ViT-B-32-multilingual-v1')
clip_b32_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
multilingual_text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
multilingual_image_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')

class MultilingualCLIP(CLIP):
    def __init__(self, *args):
        super().__init__(*args)
        self.device = 'cuda'
        self.initilize_multilingual_encoder()
        self.train(True)
        self.to(self.device)
        self.eps = 1e-2 * 1.5

    def initilize_multilingual_encoder(self):
        self.vision_encoder  = clip_b32_model.vision_model.to(self.device)
        self.vision_proj = clip_b32_model.visual_projection.to(self.device)
        self.text_encoder = multilingual_text_model.to(self.device)
        # for param in self.vision_encoder.embeddings.parameters():
        #     param.requires_grad = False
        # for param in self.text_encoder[0].auto_model.embeddings.parameters():
        #     param.requires_grad = False
        # for i in range(0, 2):
        #     for param in self.text_encoder[0].auto_model.transformer.layer[i].parameters():
        #         param.requires_grad = False
        class TrickTokenize:
            def __call__(self, text, context_length=None):
                if type(text) == str:
                    text = [text]
                self.x = text
                return self
            def to(self, device=None):
                return self.x
        def encode_text(texts, return_dense=None):
            if return_dense:
                return self.text_encoder.encode(texts, convert_to_tensor=True, device=self.device), None
            else:
                return self.text_encoder.encode(texts, convert_to_tensor=True, device=self.device)
        self.encode_text = encode_text
        self.tokenize = TrickTokenize() # do nothing because tokenize is done in encode_text
        
    def encode_image(self, image, return_dense=False):
        if return_dense:
            return self.vision_proj(self.vision_encoder(image.to(self.device)).pooler_output), None
        else:
            return self.vision_proj(self.vision_encoder(image.to(self.device)).pooler_output)

        
model = MultilingualCLIP(config, None, None, num_classes, config.experiment.ritc_eps)
config.schedule.niter_per_ep = len(train_loader)
lr_schedule = cosine_scheduler(config)
optimizer = build_optimizer(config, model)
scaler = torch.cuda.amp.GradScaler()
type(lr_schedule), type(optimizer), type(scaler)

os.environ['WANDB_NOTEBOOK_NAME'] = 'TBPS-CLIP_training.ipynb'
wandb.login()
wandb.finish()
run = wandb.init(
    project="TBPS-CLIP_experiment_14_11",
    config=config,
    name="training_" + '2',
)

import torch
import torch.nn.functional as F
@torch.no_grad()
def test(model, data_loader, max_length, device):
    tokenize = model.tokenize
    # switch to evaluate mode
    model.eval()

    dataset = data_loader.dataset
    texts = dataset.text
    num_text = len(texts)
    text_bs = 256

    text_feats = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text = tokenize(text, context_length=max_length).to(device)
        text_feat = F.normalize(model.encode_text(text), dim=-1)
        text_feats.append(text_feat)
    text_feats = torch.cat(text_feats, dim=0)

    image_feats = []
    for image in data_loader:
        image = image.to(device)
        image_feat = F.normalize(model.encode_image(image), dim=-1)
        image_feats.append(image_feat)
    image_feats = torch.cat(image_feats, dim=0)

    sims_matrix = text_feats @ image_feats.t()
    eval_result = metric_eval(sims_matrix, dataset.img2person, dataset.txt2person)

    return eval_result


@torch.no_grad()
def metric_eval(scores_t2i, img2person, txt2person):
    device = scores_t2i.device
    img2person = img2person.to(device)
    txt2person = txt2person.to(device)

    index = torch.argsort(scores_t2i, dim=-1, descending=True)
    pred_person = img2person[index]
    matches = (txt2person.view(-1, 1).eq(pred_person)).long()

    def acc_k(matches, k=1):
        matches_k = matches[:, :k].sum(dim=-1)
        matches_k = torch.sum((matches_k > 0))
        return 100.0 * matches_k / matches.size(0)

    # Compute metrics
    ir1 = acc_k(matches, k=1).item()
    ir5 = acc_k(matches, k=5).item()
    ir10 = acc_k(matches, k=10).item()
    ir_mean = (ir1 + ir5 + ir10) / 3

    real_num = matches.sum(dim=-1)
    tmp_cmc = matches.cumsum(dim=-1).float()
    order = torch.arange(start=1, end=matches.size(1) + 1, dtype=torch.long).to(device)
    tmp_cmc /= order
    tmp_cmc *= matches
    AP = tmp_cmc.sum(dim=-1) / real_num
    mAP = AP.mean() * 100.0

    eval_result = {'r1': ir1,
                   'r5': ir5,
                   'r10': ir10,
                   'r_mean': ir_mean,
                   'mAP': mAP.item()
                   }

    return eval_result




it = 0
logger = run
for epoch in range(config.schedule.epoch):
    start_time = time.time()
    for meter in meters.values():
        meter.reset()
    model.train()
    for i, batch in enumerate(train_loader):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[it] * param_group['ratio']
        if epoch == 0:
            alpha = config.model.softlabel_ratio * \
                min(1.0, i / len(train_loader))
        else:
            alpha = config.model.softlabel_ratio

        with torch.autocast(device_type='cuda'):
            ret = model(batch, alpha)
            loss = sum([v for k, v in ret.items() if "loss" in k])
        batch_size = batch['image'].shape[0]
        meters['loss'].update(loss.item(), batch_size)
        meters['nitc_loss'].update(ret.get('nitc_loss', 0), batch_size)
        meters['ss_loss'].update(ret.get('ss_loss', 0), batch_size)
        meters['citc_loss'].update(ret.get('citc_loss', 0), batch_size)
        meters['ritc_loss'].update(ret.get('ritc_loss', 0), batch_size)
        meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
        meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
        batch_size = batch['image'].shape[0]
        logger.log({
            'epoch': epoch, 
            'step': i,
            'lr': lr_schedule[it],
            **{k: v.avg for k, v in meters.items()}
        })       

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad()

        # if (i % 5 == 0) or (i == len(train_loader) - 1):
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        it += 1
        if (i + 1) % config.log.print_period == 0:
            info_str = f"Epoch[{epoch + 1}] Iteration[{i + 1}/{len(train_loader)}]"
            # log loss
            for k, v in meters.items():
                if v.val != 0:
                    info_str += f", {k}: {v.val:.4f}"
            info_str += f", Base Lr: {param_group['lr']:.2e}"
            print(info_str)

    end_time = time.time()
    time_per_batch = (end_time - start_time) / (i + 1)
    print("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
          .format(epoch + 1, time_per_batch, train_loader.batch_size / time_per_batch))

    eval_result = test(
        model, dataloader['test_loader'], config['experiment']['text_length'], config.device)
    rank_1, rank_5, rank_10, map = eval_result['r1'], eval_result['r5'], eval_result['r10'], eval_result['mAP']
    logger.log({
        'epoch': epoch, 
        'rank_1': rank_1,
        'rank_5': rank_5,
        'rank_10': rank_10,
        'mAP': map,
        'epoch_time': (end_time - start_time) / (epoch + 1),
    })

    print('Acc@1 {top1:.5f} Acc@5 {top5:.5f} Acc@10 {top10:.5f} mAP {mAP:.5f}'.format(top1=rank_1, top5=rank_5,
                                                                                      top10=rank_10, mAP=map))
    torch.cuda.empty_cache()
    if best_rank_1 < rank_1:
        best_rank_1 = rank_1
        best_epoch = epoch

        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
        }
        torch.save(save_obj, os.path.join(
            config.model.saved_path, 'checkpoint_best_14_11.pth'))

print(f"best Acc@1: {best_rank_1} at epoch {best_epoch + 1}")