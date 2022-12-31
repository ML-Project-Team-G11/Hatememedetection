import wandb
from hatememe.config import CFG

cfg = CFG()

config = {key:val for key, val in cfg.__class__.__dict__.items() if isinstance(val, (float, int, str, bool))}

if cfg.mode=="inference":
    api = wandb.Api()
    run = api.run(f"{cfg.wandb_entity}/{cfg.project_name.replace('_inference','')}/{cfg.run_id}")
    config = run.config
    wandb.init(project=cfg.project_name, entity=cfg.wandb_entity, name=cfg.experiment_name, config=config)
else:
    wandb.init(project=cfg.project_name, entity=cfg.wandb_entity, name=cfg.experiment_name, config=config)

log = wandb.log