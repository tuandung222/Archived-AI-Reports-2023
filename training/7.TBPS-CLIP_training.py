import os, sys
print(os.getcwd())
from pathlib import Path
ROOT_PATH = Path('../../paper_clones/TBPS-CLIP').resolve()
sys.path.append(str(ROOT_PATH))
IMAGE_PATH = Path('../../DATASET').resolve()
sys.path.append(str(IMAGE_PATH))

import os
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
from model.tbps_model import clip_vitb
from options import get_args
import nltk; nltk.download('stopwords')

config_path = ROOT_PATH/'config/config.yaml'
config = parse_config(config_path)

set_seed(config)
config['log']['print_period'] = 1
config['model']['checkpoint'] = ROOT_PATH/'checkpoint/ViT-B-16.pt'
config['anno_dir'] = ROOT_PATH/'annotation/CUHK-PEDES'
config['image_dir'] = IMAGE_PATH/'CUHK-PEDES/imgs'
config['device'] = 'cuda'
config['model']['use_gather'] = False
config['data']['batch_size'] = 25
config['model']['saved_path'] = ROOT_PATH/"checkpoint"

meters = {
    "loss": AverageMeter(),
    "nitc_loss": AverageMeter(),
    "ss_loss": AverageMeter(),
    "citc_loss": AverageMeter(),
    "ritc_loss": AverageMeter(),
    "mlm_loss": AverageMeter(),
    "id_loss": AverageMeter(),
}


dataloader = build_pedes_data(config)
train_loader = dataloader['train_loader']
test_loader = dataloader['test_loader']
num_classes = len(train_loader.dataset.person2text)

model = clip_vitb(config, num_classes).to('cuda')
model, load_result = load_checkpoint(model, config)

config.schedule.niter_per_ep = len(train_loader)
lr_schedule = cosine_scheduler(config)
optimizer = build_optimizer(config, model)
scaler = torch.cuda.amp.GradScaler()
type(lr_schedule), type(optimizer), type(scaler)

os.environ['WANDB_NOTEBOOK_NAME'] = 'TBPS-CLIP_training.ipynb'
# wandb.login()

wandb.finish()
run = wandb.init(
    project="TBPS-CLIP_experiment_9_11",
    config=config,
    name="training_" + '7',
)

def main():
    best_rank_1 = 0.0
    best_epoch = 0
    it = 0
    logger = run
    for epoch in range(config.schedule.epoch):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()
        for i, batch in enumerate(train_loader):
            # if i > 1: break
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
                **{k: v.avg for k, v in meters.items()}
            })       

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()
            optimizer.zero_grad()
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
            model, dataloader['test_loader'], 77, config.device)
        rank_1, rank_5, rank_10, map = eval_result['r1'], eval_result['r5'], eval_result['r10'], eval_result['mAP']
        logger.log({
            'epoch': epoch,
            'epoch_loss': meters['loss'].avg,
            'rank_1': rank_1,
            'rank_5': rank_5,
            'rank_10': rank_10,
            'mAP': map,
            'epoch_time': time_per_batch * len(train_loader),
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
                config.model.saved_path, 'checkpoint_best_9_11.pth'))

    print(f"best Acc@1: {best_rank_1} at epoch {best_epoch + 1}")
    
if __name__== "__main__":
    main()