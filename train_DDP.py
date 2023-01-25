import argparse
import os

import torch
import yaml
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import get_configs_of, to_device, log, synth_one_sample
from model import DiffSingerLoss
from data_utils import Dataset
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from evaluate import evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(rank,n_gpus, args, configs):
    if rank ==0:
       print("Prepare training ...")
       print("Total GPU:",n_gpus)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '60000'
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.cuda.set_device(rank)
    preprocess_config, model_config, train_config = configs
    # Get dataset
    dataset = Dataset(
        preprocess_config["path"]["train_filelist"], preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=8, 
        pin_memory=True,
        sampler=sampler,
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    model = DDP(model)
    num_param = get_param_num(model)
    Loss = DiffSingerLoss(args, preprocess_config, model_config, train_config).to(device)
    if rank ==0:
       print("Number of DiffSinger Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step_{}".format(args.model)]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        if rank == 1:
           inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                pitches = batch[8].clone()
                assert batch[8][0].shape[0] == batch[5][0].shape[0]
                # Forward
                output = model(*(batch[1:]))
                # Cal Loss
                losses = Loss(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    lr = optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0 and rank ==1:
                    losses_ = [sum(l.values()).item() if isinstance(l, dict) else l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Noise Loss: {:.4f}".format(
                        *losses_
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses, lr=lr)

                if step % synth_step == 0 and rank ==1 and step !=0 and step != args.restore_step:
                    assert batch[8][0].shape[0] == batch[5][0].shape[0], (batch[8][0].shape,batch[5][0].shape[0])

                    figs, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        args,
                        batch,
                        pitches,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                        model.module.diffusion,
                    )
                    log(
                        train_logger,
                        step,
                        figs=figs,
                        tag="Training",
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/reconstructed",
                        step=step
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/synthesized",
                        step=step
                    )

                if step % val_step == 0 and rank ==1 and step !=0 and step != args.restore_step:
                    model.eval()
                    message = evaluate(args, model, step, configs, val_logger, vocoder, losses)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0 and step !=0 and step != args.restore_step:
                    savepath = os.path.join(train_config["path"]["ckpt_path"], "{}.pth.tar".format(step), )
                    #rmpath = os.path.join(train_config["path"]["ckpt_path"], "{}.pth.tar".format(step-3*save_step), )
                    #os.system(f"rm {rmpath}")
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        savepath,
                    )

                if step >= total_step:
                    quit()
                step += 2
                outer_bar.update(2)
            if rank ==1:
               inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--path_tag", type=str, default="")
    parser.add_argument(
        "--model",
        type=str,
        choices=["naive", "aux", "shallow"],
        required=True,
        help="training model type",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)
    if args.model == "shallow":
        assert args.restore_step >= train_config["step"]["total_step_aux"]
    if args.model in ["aux", "shallow"]:
        train_tag = "shallow"
    elif args.model == "naive":
        train_tag = "naive"
    else:
        raise NotImplementedError
    path_tag = "_{}".format(args.path_tag) if args.path_tag != "" else args.path_tag
    train_config["path"]["ckpt_path"] = train_config["path"]["ckpt_path"]+"_{}{}".format(train_tag, path_tag)
    train_config["path"]["log_path"] = train_config["path"]["log_path"]+"_{}{}".format(train_tag, path_tag)
    train_config["path"]["result_path"] = train_config["path"]["result_path"]+"_{}{}".format(args.model, path_tag)
    if preprocess_config["preprocessing"]["pitch"]["pitch_type"] == "cwt":
        from utils.pitch_tools import get_lf0_cwt
        preprocess_config["preprocessing"]["pitch"]["cwt_scales"] = get_lf0_cwt(np.ones(10))[1]

    # Log Configuration
    print("\n==================================== Training Configuration ====================================")
    print(" ---> Type of Modeling:", args.model)
    print(" ---> Total Batch Size:", int(train_config["optimizer"]["batch_size"]))
    print(" ---> Use Pitch Embed:", model_config["variance_embedding"]["use_pitch_embed"])
    print(" ---> Use Energy Embed:", model_config["variance_embedding"]["use_energy_embed"])
    print(" ---> Path of ckpt:", train_config["path"]["ckpt_path"])
    print(" ---> Path of log:", train_config["path"]["log_path"])
    print(" ---> Path of result:", train_config["path"]["result_path"])
    print("================================================================================================")
    n_gpus = torch.cuda.device_count()
    mp.spawn(main, nprocs=n_gpus, args=(n_gpus,args, configs,))
    #main(args, configs)
