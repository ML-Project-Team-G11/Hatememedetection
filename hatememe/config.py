import os
import clip
from copy import deepcopy

HOME = os.environ.get("HOME")

class CFG:
    annotations_path = f"{HOME}/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations/"
    batch_size = 62
    # base_model = "ViT-B/32" #
    base_model = "ViT-L/14@336px"
    device = "cuda"
    epochs = 20
    home = os.environ.get("HOME")
    images_path = f"{HOME}/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images/img/"
    learning_rate = 0.001
    weight_decay = 0.0001
    seed = 42
    num_linear_layers = 3
    activation = "gelu"
    dropout_prob = 0.3
    log_every = 50
    # clip_value = 5


    fusion_method = "align"

    train_image_base_model=False
    train_text_base_model=False
    


    @property
    def image_transform(self):
        if hasattr(CFG, "model"):
            return self.preprocess
        self.model, preprocess = clip.load(self.base_model)
        self.model = deepcopy(self.model)
        return preprocess

    @property
    def text_transform(self):
        return lambda texts: clip.tokenize(texts, truncate=True)

    @property
    def image_feature_extractor(self):
        if hasattr(CFG, "model"):
            return self.model.encode_image
        self.model, self.preprocess = clip.load(self.base_model)
        self.model = deepcopy(self.model)
        return deepcopy(self.model.encode_image)

    @property
    def text_feature_extractor(self):
        if hasattr(CFG, "model"):
            return self.model.encode_text
        model, self.preprocess = clip.load(self.base_model)
        return deepcopy(model.encode_text)