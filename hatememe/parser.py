import argparse
from datetime import datetime
import os
import multiprocessing

import wandb

now = datetime.now()
HOME = os.environ.get("HOME")

def str2bool(v):
    """
    src: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse 
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description="Traning and evaluation script for hateful meme classification")

    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--annotations_path", default=f"{HOME}/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations/")
    parser.add_argument("--model_path", default=f"{HOME}/.cache/torch/mmf/data/datasets/hateful_memes/defaults/model/")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--base_model", default="ViT-L/14@336px", choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--home", default=HOME)
    parser.add_argument("--images_path", default=f"{HOME}/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images/img/")
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=0.0001, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_linear_layers", default=3, type=int)
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--dropout_prob", default=0.3, type=float)
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--add_memotion", type=str2bool, default=False)

    parser.add_argument("--add_linear_image_layers", type=str2bool, default=False)
    parser.add_argument("--add_linear_text_layers", type=str2bool, default=False)

    parser.add_argument("--fusion_method", default="concat", choices=["concat", "align", "cross", "mean-align"])

    parser.add_argument("--train_image_base_model", type=str2bool, default=False)
    parser.add_argument("--train_text_base_model", type=str2bool, default=False)

    parser.add_argument("--mode", default="train", choices=["train", "inference"])
    parser.add_argument("--data_split", default="train", choices=["train", "test-unseen", "dev-unseen", "test-seen", "dev-seen", "test-dev-all"])
    parser.add_argument("--eager_transform", default=False, type=str2bool)
    parser.add_argument("--num_workers", default=int(multiprocessing.cpu_count()*2/3), type=int)
    parser.add_argument("--project_name", default="hatememe")
    parser.add_argument("--wandb_entity", default="team-g11")
    parser.add_argument("--run_id", help="This should only be provided in inference mode", default=None)

    args = parser.parse_args()

    if args.mode=="train" and args.data_split!="train":
        raise ValueError("You can only use train mode with 'train' data_split")

    if args.mode=="inference" and not args.experiment_name:
        raise ValueError("You must provide experiment name when not in train mode")

    if args.mode=="inference":
        api = wandb.Api()
        run = api.run(f"{args.wandb_entity}/{args.project_name.replace('_inference','')}/{args.run_id}")
        config = run.config
        update_args(config, args)
        args.project_name = f"{args.project_name}_inference"

    if not args.experiment_name:
        args.experiment_name=f"Exp_{now}"

    args.model_path = os.path.join(args.model_path, f"{args.experiment_name}.pth")

    args.num_workers = min((args.num_workers, int(multiprocessing.cpu_count()*2/3)))

    return args

def update_args(config: dict, args):
    exempt = {"mode", "data_split", "eager_transform", "num_workers", "project_name", "wandb_entity", "model_path"}
    for key, val in config.items():
        if key not in exempt:
            setattr(args, key, val)