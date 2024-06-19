import os
import torch
from torchvision.datasets import CIFAR100

from .clip import load, tokenize


class CLIP_ITM():

    def __init__(self, visual_model_name,model_root,dataset_root):
        self.model, self.preprocess = load(name=visual_model_name,download_root=model_root)
        self.dataset = CIFAR100(root=os.path.expanduser(dataset_root),
                                download=False,
                                train=False)

    def forward(self, image_id, caption):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        image, class_id = self.dataset[image_id]

        image_input = self.preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        correct_class = self.dataset.classes[class_id]
        texts = [caption]
        for class_ in self.dataset.classes:
            if class_ != correct_class:
                texts.append(class_)

        text_tokens = tokenize(texts).to(device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        probs = (100.0 * image_features @ text_features.T).softmax(
            dim=-1).float().cpu().numpy().flatten()
        return probs[0]


def clip_itm(visual_model_name,model_root,dataset_root):
    return CLIP_ITM(visual_model_name,model_root,dataset_root)
