#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train d-vector."""

import json
from argparse import ArgumentParser
from datetime import datetime
from itertools import count
from multiprocessing import cpu_count
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import GE2EDataset, MultiEpochsDataLoader, pad_batch
from modules import DVector, GE2ELoss
from utils import CUDATimer


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("model_dir", type=str)
    parser.add_argument("-n", "--n_speakers", type=int, default=64)
    parser.add_argument("-m", "--n_utterances", type=int, default=10)
    parser.add_argument("--seg_len", type=int, default=160)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--valid_every", type=int, default=1000)
    parser.add_argument("--decay_every", type=int, default=100000)
    parser.add_argument("--batch_per_valid", type=int, default=10)
    parser.add_argument("--n_workers", type=int, default=cpu_count())
    parser.add_argument("--preload", action="store_true")
    return vars(parser.parse_args())


def infinite_iterator(dataloader, sampler=None):
    """Infinitely yield a batch of data."""
    for epoch in count():
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in iter(dataloader):
            yield batch


def train(
    data_dir,
    model_dir,
    n_speakers,
    n_utterances,
    seg_len,
    save_every,
    valid_every,
    decay_every,
    batch_per_valid,
    n_workers,
    preload,
):
    """Train a d-vector network."""

    # read and log training infos
    start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    checkpoints_path = Path(model_dir) / "checkpoints" / start_time
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    with open(Path(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    # create data loader, iterator
    dataset = GE2EDataset(
        data_dir, metadata["speakers"], n_utterances, seg_len, preload
    )
    trainset, validset = random_split(dataset, [len(dataset) - n_speakers, n_speakers])
    assert len(trainset) >= n_speakers
    assert len(validset) >= n_speakers
    print(f"[TRAIN] Use {len(trainset)} speakers for training.")
    print(f"[TRAIN] Use {len(validset)} speakers for validation.")

    # start distributed training
    world_size = torch.cuda.device_count()
    print(f"[TRAIN] Use {world_size} GPUs!")
    tmp_file = NamedTemporaryFile()
    print(f"[TRAIN] Use {tmp_file.name} as FileStore.")
    mp.spawn(
        ddp_train,
        args=(
            world_size,
            tmp_file.name,
            model_dir,
            n_speakers,
            n_utterances,
            seg_len,
            save_every,
            valid_every,
            decay_every,
            batch_per_valid,
            n_workers,
            start_time,
            checkpoints_path,
            metadata,
            trainset,
            validset,
        ),
        nprocs=world_size,
        join=True,
    )


def ddp_train(
    rank,
    world_size,
    tmp_file_path,
    model_dir,
    n_speakers,
    n_utterances,
    seg_len,
    save_every,
    valid_every,
    decay_every,
    batch_per_valid,
    n_workers,
    start_time,
    checkpoints_path,
    metadata,
    trainset,
    validset,
):
    print(f"[DDP] Running on rank {rank}.")
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        store=dist.FileStore(tmp_file_path, world_size),
    )

    train_sampler = DistributedSampler(trainset)
    train_loader = MultiEpochsDataLoader(
        trainset,
        batch_size=n_speakers,
        sampler=train_sampler,
        num_workers=n_workers,
        collate_fn=pad_batch,
        drop_last=True,
    )
    train_iter = infinite_iterator(train_loader, train_sampler)

    # build network and training tools
    dvector = DVector(dim_input=metadata["n_mels"], seg_len=seg_len).to(rank)
    ddp_dvector = DDP(dvector, device_ids=[rank])

    criterion = GE2ELoss().to(rank)
    ddp_criterion = DDP(criterion, device_ids=[rank])
    optimizer = SGD(
        list(ddp_dvector.parameters()) + list(ddp_criterion.parameters()), lr=0.01
    )
    scheduler = StepLR(optimizer, step_size=decay_every, gamma=0.5)

    train_losses, valid_losses = [], []
    batch_ms, model_ms, loss_ms, backward_ms = [], [], [], []

    if rank == 0:
        writer = SummaryWriter(Path(model_dir) / "logs" / start_time)
        valid_loader = MultiEpochsDataLoader(
            validset,
            batch_size=n_speakers,
            num_workers=n_workers,
            collate_fn=pad_batch,
            drop_last=True,
        )
        valid_iter = infinite_iterator(valid_loader)
        pbar = tqdm(total=valid_every, ncols=0, desc="Train")
        cuda_timer = CUDATimer()
    else:
        valid_loader, valid_iter, writer = None, None, None
        pbar, cuda_timer = None, None

    # start training
    for step in count(start=1):

        if rank == 0:
            cuda_timer.record("batch")
        batch = next(train_iter).to(rank)

        if rank == 0:
            cuda_timer.record("model")
        embds = ddp_dvector(batch).view(n_speakers, n_utterances, -1)

        if rank == 0:
            cuda_timer.record("loss")
        loss = ddp_criterion(embds)

        if rank == 0:
            cuda_timer.record("backward")
        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(ddp_dvector.parameters()) + list(ddp_criterion.parameters()),
            max_norm=3,
            norm_type=2.0,
        )
        dvector.embedding.weight.grad *= 0.5
        dvector.embedding.bias.grad *= 0.5
        criterion.w.grad *= 0.01
        criterion.b.grad *= 0.01

        optimizer.step()
        scheduler.step()

        if rank == 0:
            cuda_timer.record()
            elapsed_times = cuda_timer.stop()

            train_losses.append(loss.item())
            batch_ms.append(elapsed_times["batch"])
            model_ms.append(elapsed_times["model"])
            loss_ms.append(elapsed_times["loss"])
            backward_ms.append(elapsed_times["backward"])

            pbar.update(1)
            pbar.set_postfix(loss=loss.item(), grad_norm=grad_norm.item())

            if step % valid_every == 0:
                pbar.close()

                for _ in range(batch_per_valid):
                    batch = next(valid_iter).to(rank)

                    with torch.no_grad():
                        embd = ddp_dvector(batch).view(n_speakers, n_utterances, -1)
                        loss = ddp_criterion(embd)
                        valid_losses.append(loss.item())

                avg_train_loss = sum(train_losses) / len(train_losses)
                avg_valid_loss = sum(valid_losses) / len(valid_losses)
                avg_batch_ms = sum(batch_ms) / len(batch_ms)
                avg_model_ms = sum(model_ms) / len(model_ms)
                avg_loss_ms = sum(loss_ms) / len(loss_ms)
                avg_backward_ms = sum(backward_ms) / len(backward_ms)
                print(f"[DDP] Valid: loss={avg_valid_loss:.1f}, ")
                print(
                    f"[DDP] Average elapsed time: "
                    f"batch {avg_batch_ms:.1f} ms, "
                    f"model {avg_model_ms:.1f} ms, "
                    f"loss {avg_loss_ms:.1f} ms, "
                    f"backward {avg_backward_ms:.1f} ms"
                )

                writer.add_scalar("Loss/train", avg_train_loss, step)
                writer.add_scalar("Loss/valid", avg_valid_loss, step)

                writer.add_scalar("Elapsed time/batch (ms)", avg_batch_ms, step)
                writer.add_scalar("Elapsed time/model (ms)", avg_model_ms, step)
                writer.add_scalar("Elapsed time/loss (ms)", avg_loss_ms, step)
                writer.add_scalar("Elapsed time/backward (ms)", avg_backward_ms, step)
                writer.flush()

                pbar = tqdm(
                    total=step + valid_every, ncols=0, initial=step, desc="Train",
                )
                train_losses, valid_losses = [], []

            if step % save_every == 0:
                ckpt_path = checkpoints_path / f"dvector-step{step}.pt"
                torch.save(ddp_dvector, str(ckpt_path))

    dist.destroy_process_group()


if __name__ == "__main__":
    train(**parse_args())
