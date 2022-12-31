import torch
import torch.nn as nn
import torch.nn.functional as F
from hatememe.config import CFG
from math import ceil
import clip
from copy import deepcopy

from functools import cached_property

class HMMLP(nn.Module):
    def __init__(
        self,
        n_out=1,
        config:CFG = CFG(),
    ) -> None:
        super().__init__()
        print("Model initialized...")
        for attr, val in config.__class__.__dict__.items():
            if isinstance(val, (int, float, str, list, tuple, set,)):
                print(f"{attr:20}: {val}")
        self.config = config

        self.base_model = self._base_model

        self.n_in = self._get_linear_input_dim()
        self.n_out = n_out

        self.text_projection_layer = self.text_projection()
        self.image_projection_layer = self.image_projection()
        self.linear_layers = self._get_linear_layers()

        n_inp_final = 1
        if config.add_linear_image_layers:
            self.linear_image_layers = self._get_linear_layers(n_in=self.base_model.visual.output_dim)
            n_inp_final +=1
        if config.add_linear_text_layers:
            self.linear_text_layers = self._get_linear_layers(n_in=self.base_model.transformer.width)
            n_inp_final +=1

        self.output_layer = nn.Linear(n_inp_final, n_out)

        self.freeze_pretrained_weights()

    def text_projection(self):
        n_in, n_out = self.base_model.text_projection.shape
        layer = nn.Linear(n_in, n_out, bias=False)
        layer.weight = nn.Parameter(deepcopy(self.base_model.text_projection).T)
        activation = self._activation_map(self.config.activation)
        return nn.Sequential(layer, activation)

    def image_projection(self):
        n_in, n_out = self.base_model.visual.proj.shape
        layer = nn.Linear(n_in, n_out, bias=False)
        layer.weight = nn.Parameter(deepcopy(self.base_model.visual.proj).T)
        activation = self._activation_map(self.config.activation)
        return nn.Sequential(layer, activation)

    @cached_property
    def _base_model(self):
        if hasattr(self.config, "model"):
            return self.config.model
        model, _ = clip.load(self.config.base_model, device=torch.device(self.config.device))
        return deepcopy(model)

    def freeze_pretrained_weights(self):
        for param_name, p in self.base_model.named_parameters():
            if self.config.train_image_base_model and param_name.startswith("model.visual"):
                continue
            if self.config.train_text_base_model and param_name.startswith("model.transformer"):
                continue
            p.requires_grad_(False)
        

    def _get_linear_layers(self, n_in=None):
        hidden_neurons = []
        n_in = n_in or self.n_in
        activation = self._activation_map(self.config.activation)
        for i in range(self.config.num_linear_layers-1):
            n_hid = ceil(n_in*(0.25))
            hidden_neurons.extend([nn.Linear(n_in, n_hid), nn.Dropout(p=self.config.dropout_prob), activation])
            n_in = n_hid
            if n_in<=32:
                break

        hidden_neurons.append(nn.Linear(n_in, self.n_out))

        return nn.Sequential(*hidden_neurons)

    def _get_linear_input_dim(self,):
        map = {
            "concat":self.base_model.visual.output_dim*2, 
            "cross":self.base_model.visual.output_dim**2,
            "align":self.base_model.visual.output_dim,
            "mean-align":self.base_model.visual.output_dim,
        }

        return map[self.config.fusion_method]
    
    def _activation_map(self, activation_name):
        map = {"relu":nn.ReLU(), "gelu":nn.GELU(), "sigmoid":nn.Sigmoid()}
        if activation_name not in map:
            raise ValueError("Unrecognized/unconfigured activation specified")
        activation_function = map[activation_name]
        return activation_function


    def _fusion_method_map(self, method):
        map = {
            "concat": lambda images, texts: torch.hstack((images, texts)),
            "cross": lambda images, texts: torch.bmm(images.unsqueeze(dim=2), texts.unsqueeze(dim=1)).view(images.shape[0], -1),
            "align": lambda images, texts: torch.mul(images, texts),
            "mean-align": lambda images, texts: torch.divide(torch.add(images, texts), torch.tensor(2))
        }
        if callable(method):
            return method
        fusion_callable = map.get(method)
        if fusion_callable:
            return fusion_callable
        raise ValueError(f"Invalid fusion method `{method}` provided.")

    def fuse_image_text_embd(self, images, texts, fusion_method):
        fusion_function = self._fusion_method_map(fusion_method)
        fused_images_texts = fusion_function(images, texts)
        return fused_images_texts

    def pre_output_fusion(self, *args):
        return torch.hstack(args)

    def forward(self, images, texts):

        images = self.base_model.encode_image(images)
        texts = self.base_model.encode_text(texts.squeeze())

        # Apply the projection layers
        images = self.image_projection_layer(images)
        texts = self.text_projection_layer(texts)
        
        # Normalize the outputs
        images = images / images.norm(dim=1, keepdim=True)
        texts = texts / texts.norm(dim=1, keepdim=True)

        images_texts_fused = self.fuse_image_text_embd(images, texts, self.config.fusion_method)

        images_texts_logits = self.linear_layers(images_texts_fused)

        args_to_pre_output=[images_texts_logits]
        if self.config.add_linear_image_layers:
            images_logits = self.linear_image_layers(images)
            args_to_pre_output.append(images_logits)
        if self.config.add_linear_text_layers:
            texts_logits = self.linear_text_layers(texts)
            args_to_pre_output.append(texts_logits)

        pre_output = self.pre_output_fusion(*args_to_pre_output)

        logits = self.output_layer(pre_output)

        return logits