import os
import torch
from torchvision.datasets import CIFAR100

from .clip import load, tokenize


class CLIP_CLF():

    def __init__(self, visual_model_name,model_root,dataset_root):
        self.model, self.preprocess = load(name=visual_model_name,download_root=model_root)
        self.dataset = CIFAR100(root=dataset_root,
                                download=False,
                                train=False)

    def forward(self, image_id, class_list):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        image, _ = self.dataset[image_id]
        image_input = self.preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        text_descriptions = [
            f"This is a photo of a {class_}" for class_ in class_list
        ]
        # text_descriptions = ["a photo of dog", "a photo of a snake"]
        text_tokens = tokenize(text_descriptions).to(device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        probs = (100.0 * image_features @ text_features.T).softmax(
            dim=-1).float().cpu().numpy()
        return probs


def clip_clf(visual_model_name,model_root,dataset_root):
    return CLIP_CLF(visual_model_name,model_root,dataset_root)
