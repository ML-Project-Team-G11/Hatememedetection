import wandb
from hatememe.config import CFG
from datetime import datetime

now = datetime.now()

cfg = CFG()

wandb.init(project="hatememe", entity="team-g11", name=f"Experiment_{now}")

wandb.config = CFG.__class__.__dict__

log = wandb.log