import os
import sys
from torchvision.datasets import CIFAR100

sys.path.append(os.path.join(os.getcwd()))

from models import clip_itm

visual_model = "ViT-B/16"
image_id = 3637
model_root="/workspace/pretrained"
dataset_root="/workspace/datasets"
cifar100 = CIFAR100(root=dataset_root, download=False, train=False)
_, class_id = cifar100[image_id]
correct_class = cifar100.classes[class_id]
# caption = f"This is a photo of a {correct_class}"
caption = "This is a photo of a dog"

model = clip_itm(visual_model,model_root,dataset_root)
results = model.forward(image_id, caption)
print(results)
