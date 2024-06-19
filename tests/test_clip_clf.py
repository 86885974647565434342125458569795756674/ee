import os
import sys
import random
from torchvision.datasets import CIFAR100

sys.path.append(os.path.join(os.getcwd()))

from models import clip_clf

K = 10

visual_model = "ViT-B/16"
image_id = 3637
dataset_root="/workspace/datasets"
model_root="/workspace/pretrained"
cifar100 = CIFAR100(root=dataset_root, download=False, train=False)
_, class_id = cifar100[image_id]
correct_class = cifar100.classes[class_id]
class_list = [correct_class]
while len(class_list) < K:
    random_class = random.choice(cifar100.classes)
    if random_class != correct_class:
        class_list.append(random_class)
print(class_list)

model = clip_clf(visual_model,model_root,dataset_root)
results = model.forward(image_id, class_list)
for result in results:
    print([f"{c:.4f}" for c in result])
