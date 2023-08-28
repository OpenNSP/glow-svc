import logging
import multiprocessing
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import modules.commons as commons
import utils
from data_utils import TextAudioCollate, TextAudioSpeakerLoader
from diffusion.vocoder import Vocoder
from models import (
    SynthesizerTrn,
)
from modules.losses import mle_loss
from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    hps = utils.get_hparams()

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = hps.train.port

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    
    # for pytorch on win, backend use gloo    
    dist.init_process_group(backend=  'gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    collate_fn = TextAudioCollate()
    all_in_mem = hps.train.all_in_mem   # If you have enough memory, turn on this option to avoid disk IO and speed up training.
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps, all_in_mem=all_in_mem)
    num_workers = 5 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()
    if all_in_mem:
        num_workers = 0
    train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=False, pin_memory=True,
                              batch_size=hps.train.batch_size, collate_fn=collate_fn)
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps, all_in_mem=all_in_mem,vol_aug = False)
        eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
                                 batch_size=1, pin_memory=False,
                                 drop_last=False, collate_fn=collate_fn)
    
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda(rank)
    optim_g = commons.Adam(
        net_g.parameters(),
        scheduler=hps.train.scheduler,
        dim_model=hps.model.hidden_channels,
        warmup_steps=hps.train.warmup_steps,
        lr=hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)  # , find_unused_parameters=True)
    if rank == 0:
        vocoder = Vocoder('nsf-hifigan','pretrain/nsf_hifigan/model', device="cuda")
    skip_optimizer = False
    try:
        name = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
        _, _, _, epoch_str = utils.load_checkpoint(name, net_g,
                                                   optim_g, skip_optimizer)
        epoch_str = max(epoch_str, 1)
        global_step=int(name[name.rfind("_")+1:name.rfind(".")])+1
        optim_g.step_num = global_step
        optim_g._update_learning_rate()
        #global_step = (epoch_str - 1) * len(train_loader)
    except Exception:
        print("load old checkpoint failed...")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0
    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        # set up warm-up learning rate
        # if epoch <= warmup_epoch:
        #     for param_group in optim_g.param_groups:
        #         param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
        # training
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, net_g, optim_g, scaler,
                               [train_loader, eval_loader], logger, [writer, writer_eval],vocoder)
        else:
            train_and_evaluate(rank, epoch, hps, net_g, optim_g, scaler,
                               [train_loader, None], None, None,vocoder)


def train_and_evaluate(rank, epoch, hps, nets, optims, scaler, loaders, logger, writers, vocoder):
    image_dict = {}
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers
    
    half_type = torch.bfloat16 if hps.train.half_type=="bf16" else torch.float16

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    nets.train()
    for batch_idx, items in enumerate(train_loader):
        c, f0, spec, y, spk, lengths, uv,volume = items
        g = spk.cuda(rank, non_blocking=True)
        spec, y = spec.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)
        c = c.cuda(rank, non_blocking=True)
        f0 = f0.cuda(rank, non_blocking=True)
        uv = uv.cuda(rank, non_blocking=True)
        lengths = lengths.cuda(rank, non_blocking=True)
        
        with autocast(enabled=hps.train.fp16_run, dtype=half_type):
            x_mask, (z_p, m_p, logs_p), logdet, pred_lf0, norm_lf0, lf0 = nets(c, f0, uv, spec, g=g, c_lengths=lengths,
                                                                                vol = volume)

            with autocast(enabled=False, dtype=half_type):
                loss_mle = mle_loss(z_p,m_p,logs_p,logdet,x_mask)
        
        optims.zero_grad()
        scaler.scale(loss_mle).backward()
        grad_norm = commons.clip_grad_value_(nets.parameters(), 5)
        scaler.unscale_(optims._optim)
        scaler.step(optims._optim)
        optims._update_learning_rate()
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optims.get_lr()
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info(f"Losses: {loss_mle}, step: {global_step}, lr: {lr}, ,grad_norm: {grad_norm}")

                scalar_dict = {"loss_mle": loss_mle, "lr": lr, "grad_norm": grad_norm}

                # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})

                if nets.module.use_automatic_f0_prediction:
                    image_dict = ({
                        "all/lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                              pred_lf0[0, 0, :].detach().cpu().numpy()),
                        "all/norm_lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                                   norm_lf0[0, 0, :].detach().cpu().numpy())
                    })

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict
                )

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, nets, eval_loader, writer_eval, vocoder)
                utils.save_checkpoint(nets, optims._optim, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)

        global_step += 1

    if rank == 0:
        global start_time
        now = time.time()
        durtaion = format(now - start_time, '.2f')
        logger.info(f'====> Epoch: {epoch}, cost {durtaion} s')
        start_time = now


def evaluate(hps, generator, eval_loader, writer_eval, vocoder):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            c, f0, spec, y, spk, _, uv,volume = items
            g = spk[:1].cuda(0)
            spec, y = spec[:1].cuda(0), y[:1].cuda(0)
            c = c[:1].cuda(0)
            f0 = f0[:1].cuda(0)
            uv= uv[:1].cuda(0)
            if volume is not None:
                volume = volume[:1].cuda(0)

            y_mel,_ = generator.module.infer(c, f0, uv, g=g,vol = volume)
            _f0 = f0.transpose(-1, -2).unsqueeze(0)
            _mel = y_mel.transpose(-1, -2)
            #print(f0.shape)
            #print(y_mel.shape)
            y_hat = vocoder.infer(_mel, _f0).squeeze()
           
            audio_dict.update({
                f"gen/audio_{batch_idx}": y_hat,
                f"gt/audio_{batch_idx}": y[0]
            })
        image_dict.update({
            "gen/mel": utils.plot_spectrogram_to_numpy(y_mel[0].cpu().numpy()),
            "gt/mel": utils.plot_spectrogram_to_numpy(spec[0].cpu().numpy())
        })
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
    main()