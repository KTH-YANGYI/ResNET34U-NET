import os
import pickle

import torch

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
except Exception:
    dist = None
    DistributedDataParallel = None


def unwrap_model(model):
    if DistributedDataParallel is not None and isinstance(model, DistributedDataParallel):
        return model.module
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def model_state_dict(model):
    return unwrap_model(model).state_dict()


def env_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def env_rank():
    return int(os.environ.get("RANK", "0"))


def env_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def distributed_requested(cfg):
    return bool(cfg.get("ddp", False) or cfg.get("distributed", False) or env_world_size() > 1)


def init_distributed(cfg):
    requested = distributed_requested(cfg)
    world_size = env_world_size()
    context = {
        "distributed": False,
        "requested": requested,
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "backend": "",
        "object_group": None,
    }

    if not requested or world_size <= 1:
        return context

    if dist is None or not dist.is_available():
        raise RuntimeError("torch.distributed is not available, but DDP was requested")

    backend = str(cfg.get("ddp_backend", "nccl" if torch.cuda.is_available() else "gloo")).strip().lower()
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    local_rank = env_local_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    object_group = None
    if backend == "nccl":
        object_group = dist.new_group(backend="gloo")

    context.update(
        {
            "distributed": True,
            "rank": int(rank),
            "local_rank": int(local_rank),
            "world_size": int(world_size),
            "backend": backend,
            "object_group": object_group,
        }
    )
    return context


def distributed_device(cfg, context):
    if context.get("distributed", False):
        if torch.cuda.is_available():
            return torch.device("cuda", int(context["local_rank"]))
        return torch.device("cpu")

    device_text = str(cfg.get("device", "auto")).strip().lower()
    if device_text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_text == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(device_text)


def is_main_process(context=None):
    return context is None or int(context.get("rank", 0)) == 0


def barrier(context):
    if context is not None and context.get("distributed", False):
        dist.barrier()


def cleanup_distributed(context):
    if context is not None and context.get("distributed", False) and dist is not None and dist.is_initialized():
        object_group = context.get("object_group")
        if object_group is not None:
            dist.destroy_process_group(object_group)
        dist.destroy_process_group()


def local_batch_size(global_batch_size, context):
    if context is None or not context.get("distributed", False):
        return max(1, int(global_batch_size))

    world_size = max(1, int(context.get("world_size", 1)))
    return max(1, (int(global_batch_size) + world_size - 1) // world_size)


def local_batch_size_map(batch_size_by_key, context):
    return {
        str(key): local_batch_size(value, context)
        for key, value in dict(batch_size_by_key or {}).items()
    }


def maybe_wrap_ddp(model, cfg, device, context):
    summary = {
        "ddp": False,
        "requested": distributed_requested(cfg),
        "device_type": str(device.type),
        "rank": int(context.get("rank", 0)),
        "local_rank": int(context.get("local_rank", 0)),
        "world_size": int(context.get("world_size", 1)),
        "backend": context.get("backend", ""),
    }

    if not context.get("distributed", False):
        return model, summary

    if DistributedDataParallel is None:
        raise RuntimeError("DistributedDataParallel is not available")

    find_unused = bool(cfg.get("ddp_find_unused_parameters", True))
    if device.type == "cuda":
        wrapped = DistributedDataParallel(
            model,
            device_ids=[int(context["local_rank"])],
            output_device=int(context["local_rank"]),
            find_unused_parameters=find_unused,
        )
    else:
        wrapped = DistributedDataParallel(model, find_unused_parameters=find_unused)

    summary.update({"ddp": True, "find_unused_parameters": find_unused})
    return wrapped, summary


def broadcast_object(value, context, src=0):
    if context is None or not context.get("distributed", False):
        return value

    object_group = context.get("object_group")
    if object_group is not None or context.get("backend", "") != "nccl":
        payload = [value]
        dist.broadcast_object_list(payload, src=int(src), group=object_group)
        return payload[0]

    device = torch.device("cpu")
    if torch.cuda.is_available() and context.get("backend", "") == "nccl":
        device = torch.device("cuda", int(context.get("local_rank", 0)))

    rank = int(context.get("rank", 0))
    if rank == int(src):
        payload = pickle.dumps(value)
        length = torch.tensor([len(payload)], dtype=torch.long, device=device)
    else:
        payload = b""
        length = torch.tensor([0], dtype=torch.long, device=device)

    dist.broadcast(length, src=int(src))
    payload_size = int(length.item())

    if rank == int(src):
        data = torch.tensor(list(payload), dtype=torch.uint8, device=device)
    else:
        data = torch.empty((payload_size,), dtype=torch.uint8, device=device)

    if payload_size > 0:
        dist.broadcast(data, src=int(src))

    if rank == int(src):
        return value

    return pickle.loads(bytes(data.cpu().tolist()))


def sync_train_stats(stats, device, context):
    if context is None or not context.get("distributed", False):
        return stats

    sample_count = int(stats.get("sample_count", 0))
    values = torch.tensor(
        [float(stats["loss"]) * sample_count, float(sample_count)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total_samples = max(1.0, float(values[1].item()))
    synced = dict(stats)
    synced["loss"] = float(values[0].item() / total_samples)
    synced["sample_count"] = int(values[1].item())
    return synced


def parse_device_ids(value, available_count):
    if value is None or value == "":
        return list(range(int(available_count)))

    if isinstance(value, int):
        return [int(value)]

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return [int(part) for part in parts if part != ""]

    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]

    raise ValueError("data_parallel_device_ids must be an int, list, or comma-separated string")


def maybe_wrap_data_parallel(model, cfg, device):
    enabled = bool(cfg.get("data_parallel", False) or cfg.get("multi_gpu", False))
    summary = {
        "data_parallel": False,
        "requested": enabled,
        "device_type": str(device.type),
        "available_cuda_devices": 0,
        "device_ids": [],
    }

    if not enabled or device.type != "cuda":
        return model, summary

    available_count = int(torch.cuda.device_count())
    summary["available_cuda_devices"] = available_count
    if available_count < 2:
        return model, summary

    device_ids = parse_device_ids(cfg.get("data_parallel_device_ids", ""), available_count)
    device_ids = [device_id for device_id in device_ids if 0 <= int(device_id) < available_count]
    if len(device_ids) < 2:
        summary["device_ids"] = device_ids
        return model, summary

    output_device = int(device_ids[0])
    wrapped_model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device)
    summary.update(
        {
            "data_parallel": True,
            "device_ids": device_ids,
            "output_device": output_device,
        }
    )
    return wrapped_model, summary
