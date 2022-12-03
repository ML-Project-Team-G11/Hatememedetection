import wandb
from hatememe.config import CFG

cfg = CFG()

config = {key:val for key, val in cfg.__class__.__dict__.items() if isinstance(val, (float, int, str, bool))}

wandb.init(project="hatememe", entity="team-g11", name=config.experiment_name, config=config)

log = wandb.log