from pathlib import Path


def _normalize_mode(mode):
    mode = str(mode).strip().lower()
    if mode in {"online", "offline", "disabled"}:
        return mode
    return "online"


def _sanitize_config(value):
    if isinstance(value, dict):
        return {str(key): _sanitize_config(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_sanitize_config(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    return value


def init_wandb_run(cfg, stage_name, fold, save_dir):
    if not bool(cfg.get("wandb_enable", True)):
        return None

    try:
        import wandb
    except Exception as exc:
        print(f"Warning: wandb import failed, training will continue without wandb. Detail: {exc}")
        return None

    stage_name = str(stage_name).strip()
    mode = _normalize_mode(cfg.get("wandb_mode", "online"))
    project = str(cfg.get("wandb_project", "unet-two-stage")).strip()
    entity = str(cfg.get("wandb_entity", "")).strip() or None
    group = str(cfg.get("wandb_group", "two-stage-baseline")).strip() or None
    job_type = str(cfg.get("wandb_job_type", stage_name)).strip() or None
    notes = str(cfg.get("wandb_notes", "")).strip() or None
    tags = [str(item).strip() for item in cfg.get("wandb_tags", []) if str(item).strip() != ""]

    run_name_template = str(cfg.get("wandb_run_name_template", "{stage}_fold{fold}")).strip()
    run_name = run_name_template.format(stage=stage_name, fold=int(fold))
    run_dir = Path(save_dir) / "wandb"
    run_dir.mkdir(parents=True, exist_ok=True)

    init_kwargs = {
        "project": project,
        "entity": entity,
        "group": group,
        "job_type": job_type,
        "name": run_name,
        "notes": notes,
        "tags": tags,
        "config": _sanitize_config(dict(cfg)),
        "dir": str(run_dir),
        "mode": mode,
        "reinit": True,
    }

    try:
        return wandb.init(**init_kwargs)
    except Exception as exc:
        if mode == "online":
            print(f"Warning: wandb online init failed, falling back to offline mode. Detail: {exc}")
            try:
                init_kwargs["mode"] = "offline"
                return wandb.init(**init_kwargs)
            except Exception as offline_exc:
                print(f"Warning: wandb offline init also failed, training will continue without wandb. Detail: {offline_exc}")
                return None

        print(f"Warning: wandb init failed, training will continue without wandb. Detail: {exc}")
        return None


def log_wandb_metrics(run, metrics, step=None):
    if run is None:
        return

    payload = _sanitize_config(metrics)
    if step is None:
        run.log(payload)
    else:
        run.log(payload, step=int(step))


def update_wandb_summary(run, metrics):
    if run is None:
        return

    payload = _sanitize_config(metrics)
    for key, value in payload.items():
        run.summary[key] = value


def finish_wandb_run(run):
    if run is None:
        return

    try:
        run.finish()
    except Exception:
        return
