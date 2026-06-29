# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py  # noqa: E501
import functools
import os
import subprocess
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _prepare_dist_init_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Map dist_params / env to ``init_process_group`` keyword arguments."""
    prepared = dict(kwargs)
    # MASTER_PORT is set by torchrun / slurm bootstrap, not init_process_group.
    prepared.pop("port", None)

    timeout_min = prepared.pop("timeout_min", None)
    if timeout_min is None:
        env_timeout = os.environ.get("TRAIN_DIST_TIMEOUT_MIN")
        if env_timeout is not None and str(env_timeout).strip():
            timeout_min = float(env_timeout)
    if timeout_min is not None:
        prepared["timeout"] = timedelta(minutes=float(timeout_min))

    return prepared


def init_dist(launcher, backend="nccl", **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    if launcher == "pytorch":
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == "slurm":
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f"Invalid launcher type: {launcher}")


def _init_dist_pytorch(backend, **kwargs):
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    init_kwargs = _prepare_dist_init_kwargs(kwargs)
    if rank == 0:
        timeout = init_kwargs.get("timeout")
        print(
            f"init_process_group backend={backend} timeout={timeout or 'default'}",
            flush=True,
        )
    dist.init_process_group(backend=backend, **init_kwargs)


def _init_dist_slurm(backend, port=None, **kwargs):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    # specify master port
    if port is not None:
        os.environ["MASTER_PORT"] = str(port)
    elif "MASTER_PORT" in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ["MASTER_PORT"] = "29500"
    os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
    os.environ["RANK"] = str(proc_id)
    dist.init_process_group(backend=backend, **_prepare_dist_init_kwargs(kwargs))


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
