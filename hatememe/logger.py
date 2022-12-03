import wandb
from hatememe.config import CFG
from datetime import datetime

now = datetime.now()

cfg = CFG()

config = {key:val for key, val in cfg.__class__.__dict__.items() if isinstance(val, (float, int, str, bool))}

wandb.init(project="hatememe", entity="team-g11", name=f"Exp_{now}", config=config)

log = wandb.log