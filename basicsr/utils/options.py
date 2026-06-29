import argparse
import json
import os
import random
import sys
import time
from collections import OrderedDict
from os import path as osp
from pathlib import Path

import torch
import yaml

from basicsr.utils import set_random_seed
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only

DEBUG_LOG = Path(__file__).resolve().parents[2] / ".cursor" / "debug-c42c65.log"


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    payload = {
        "sessionId": "c42c65",
        "runId": os.environ.get("CI_PIPELINE_ID", "local"),
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    line = json.dumps(payload, ensure_ascii=True)
    print(f"[debug] {line}", flush=True)
    try:
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    # #endregion


def _strip_local_rank_argv(argv: list[str]) -> tuple[list[str], int | None]:
    """Remove torch.distributed.launch / torchrun rank flags before argparse."""
    filtered: list[str] = []
    local_rank: int | None = None
    index = 0
    while index < len(argv):
        token = argv[index]
        if token.startswith("--local-rank="):
            local_rank = int(token.split("=", 1)[1])
            index += 1
            continue
        if token == "--local-rank":
            if index + 1 >= len(argv):
                raise ValueError("Missing value for --local-rank")
            local_rank = int(argv[index + 1])
            index += 2
            continue
        filtered.append(token)
        index += 1
    return filtered, local_rank


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, "r") as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = "\n"
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_level * 2) + k + ":["
            msg += dict2str(v, indent_level + 1)
            msg += " " * (indent_level * 2) + "]\n"
        else:
            msg += " " * (indent_level * 2) + k + ": " + str(v) + "\n"
    return msg


def _postprocess_yml_value(value):
    # None
    if value == "~" or value.lower() == "none":
        return None
    # bool
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    # !!float number
    if value.startswith("!!float"):
        return float(value.replace("!!float", ""))
    # number
    if value.isdigit():
        return int(value)
    elif value.replace(".", "", 1).isdigit() and value.count(".") < 2:
        return float(value)
    # list
    if value.startswith("["):
        return eval(value)
    # str
    return value


def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, required=True, help="Path to option YAML file.")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"], default="none", help="job launcher")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument(
        "--force_yml", nargs="+", default=None, help="Force to update yml files. Examples: train:ema_decay=0.999"
    )
    argv_tail, local_rank_argv = _strip_local_rank_argv(sys.argv[1:])
    _agent_log(
        "A",
        "options.py:parse_options",
        "argv before parse_args",
        {
            "raw_argv": sys.argv[1:],
            "filtered_argv": argv_tail,
            "local_rank_argv": local_rank_argv,
            "env_local_rank": os.environ.get("LOCAL_RANK"),
        },
    )
    args = parser.parse_args(argv_tail)
    if local_rank_argv is not None:
        args.local_rank = local_rank_argv
    elif os.environ.get("LOCAL_RANK") is not None:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    _agent_log(
        "B",
        "options.py:parse_options",
        "resolved local rank",
        {"local_rank": args.local_rank, "launcher": args.launcher},
    )

    # parse yml to dict
    opt = yaml_load(args.opt)

    if args.launcher == "none":
        opt["dist"] = False
        print("Disable distributed.", flush=True)
    else:
        opt["dist"] = True

    # force to update yml options (before dist init)
    if args.force_yml is not None:
        for entry in args.force_yml:
            # now do not support creating new keys
            keys, value = entry.split("=")
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = "opt"
            for key in keys.split(":"):
                eval_str += f'["{key}"]'
            eval_str += "=value"
            # using exec function
            exec(eval_str)

    opt["auto_resume"] = args.auto_resume
    opt["is_train"] = is_train

    # debug setting
    if args.debug and not opt["name"].startswith("debug"):
        opt["name"] = "debug_" + opt["name"]

    if opt["num_gpu"] == "auto":
        opt["num_gpu"] = torch.cuda.device_count()

    # datasets
    for phase, dataset in opt["datasets"].items():
        # for multiple datasets, e.g., val_1, val_2; test_1, test_2
        phase = phase.split("_")[0]
        dataset["phase"] = phase
        if "scale" in opt:
            dataset["scale"] = opt["scale"]
        if dataset.get("dataroot_gt") is not None:
            dataset["dataroot_gt"] = osp.expanduser(dataset["dataroot_gt"])
        if dataset.get("dataroot_lq") is not None:
            dataset["dataroot_lq"] = osp.expanduser(dataset["dataroot_lq"])

    # paths
    for key, val in opt["path"].items():
        if (val is not None) and ("resume_state" in key or "pretrain_network" in key):
            opt["path"][key] = osp.expanduser(val)

    if is_train:
        experiments_root = opt["path"].get("experiments_root")
        if experiments_root is None:
            experiments_root = osp.join(root_path, "experiments")
        experiments_root = osp.join(experiments_root, opt["name"])

        opt["path"]["experiments_root"] = experiments_root
        opt["path"]["models"] = osp.join(experiments_root, "models")
        opt["path"]["training_states"] = osp.join(experiments_root, "training_states")
        opt["path"]["log"] = experiments_root
        opt["path"]["visualization"] = osp.join(experiments_root, "visualization")

        # change some options for debug mode
        if "debug" in opt["name"]:
            if opt.get("val") is not None:
                opt["val"]["val_freq"] = 8
            opt["logger"]["print_freq"] = 1
            opt["logger"]["save_checkpoint_freq"] = 8
    else:  # test
        results_root = opt["path"].get("results_root")
        if results_root is None:
            results_root = osp.join(root_path, "results")
        results_root = osp.join(results_root, opt["name"])

        opt["path"]["results_root"] = results_root
        opt["path"]["log"] = results_root
        opt["path"]["visualization"] = osp.join(results_root, "visualization")

    # Re-apply force_yml so CLI overrides win over debug defaults (e.g. val:val_freq).
    if args.force_yml is not None:
        for entry in args.force_yml:
            keys, value = entry.split("=")
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = "opt"
            for key in keys.split(":"):
                eval_str += f'["{key}"]'
            eval_str += "=value"
            exec(eval_str)

    if opt["dist"]:
        dist_params = opt.get("dist_params") or {}
        init_dist(args.launcher, **dist_params)

    opt["rank"], opt["world_size"] = get_dist_info()

    seed = opt.get("manual_seed")
    if seed is None:
        seed = random.randint(1, 10000)
        opt["manual_seed"] = seed
    set_random_seed(seed + opt["rank"])

    return opt, args


@master_only
def copy_opt_file(opt_file, experiments_root):
    # copy the yml file to the experiment root
    import sys
    import time
    from shutil import copyfile

    cmd = " ".join(sys.argv)
    filename = osp.join(experiments_root, osp.basename(opt_file))
    copyfile(opt_file, filename)

    with open(filename, "r+") as f:
        lines = f.readlines()
        lines.insert(0, f"# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n")
        f.seek(0)
        f.writelines(lines)
