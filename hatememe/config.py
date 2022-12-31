import clip
from copy import deepcopy

from hatememe import parser

args = parser.get_args()

class CFG:
    experiment_name = args.experiment_name
    annotations_path = args.annotations_path
    model_path = args.model_path
    batch_size = args.batch_size
    base_model = args.base_model
    device = args.device
    epochs = args.epochs
    home = args.home
    images_path = args.images_path
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    seed = args.seed
    num_linear_layers = args.num_linear_layers
    activation = args.activation
    dropout_prob = args.dropout_prob
    log_every = args.log_every
    add_memotion = args.add_memotion

    add_linear_image_layers = args.add_linear_image_layers
    add_linear_text_layers = args.add_linear_text_layers

    fusion_method = args.fusion_method

    train_image_base_model = args.train_image_base_model
    train_text_base_model = args.train_text_base_model

    mode = args.mode
    data_split = args.data_split    

    eager_transform = args.eager_transform

    num_workers = args.num_workers
    project_name = args.project_name
    wandb_entity = args.wandb_entity
    run_id = args.run_id


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

