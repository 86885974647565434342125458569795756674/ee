import numpy as np
import torch
import os
import sys
sys.path.append(os.path.join(os.getcwd()))
from models.blip.blip_nlvr_text_encoder import blip_nlvr_text_encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


images_embeds = np.load("/workspace/pretrained/blip_nlvr_images_embeds.npy")

questions = np.array(
    [b"The left image contains twice the number of dogs as the right image, and at least two dogs in total are standing.",
    b"One image shows exactly two brown acorns in back-to-back caps on green foliage.",]
)


model_url = "/workspace/pretrained/model_base_nlvr.pth"

model = blip_nlvr_text_encoder(pretrained=model_url, vit="base")

model.eval()
print(sum(p.numel() for p in model.parameters()))

model = model.to(device)

with torch.no_grad():
    questions_states = model(images_embeds, questions)
    
print(questions_states.shape,questions_states.dtype)
with open("/workspace/pretrained/blip_nlvr_questions_states.npy", "wb") as f:
    np.save(f, questions_states)


