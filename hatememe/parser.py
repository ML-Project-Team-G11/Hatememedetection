import argparse
from datetime import datetime
import os

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

    parser.add_argument("--experiment_name", default=f"Exp_{now}")
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


    parser.add_argument("--fusion_method", default="concat", choices=["concat", "align", "cross"])

    parser.add_argument("--train_image_base_model", type=str2bool, default=False)
    parser.add_argument("--train_text_base_model", type=str2bool, default=False)

    args = parser.parse_args()

    args.model_path = f"{args.model_path}/{args.experiment_name}.pth"

    return args