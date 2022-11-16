import os

HOME = os.environ.get("HOME")

class CFG:
    annotations_path = f"{HOME}/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations/"
    batch_size = 32
    clip_model_type = "ViT-B/32"
    device = "cuda"
    epochs = 10
    home = os.environ.get("HOME")
    images_path = f"{HOME}/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images/img/"
    learning_rate = 0.001
    seed = 2022
    


    @property
    def image_transform(self):
        import clip
        _, preprocess = clip.load(self.clip_model_type)
        return preprocess

    @property
    def text_transform(self):
        import clip
        return lambda texts: clip.tokenize(texts, truncate=True)

    @property
    def image_feature_extractor(self):
        import clip
        model, _ = clip.load(self.clip_model_type)
        return model.encode_image

    @property
    def text_feature_extractor(self):
        import clip
        model, _ = clip.load(self.clip_model_type)
        print("# INFO: Call ")
        return model.encode_text