from models.blip.med import BertConfig, BertModel, BertLMHeadModel
from models.blip.blip import create_vit, init_tokenizer, load_checkpoint
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
import time


class BLIP_VQA_VISUAL_ENCODER(nn.Module):
    def __init__(
        self,
        med_config="configs/med_config.json",
        image_size=480,
        vit="base",
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.image_size=image_size
        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1
        )

    def forward(self, image_urls):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print("batch size:", image_urls.shape)

        # Visual Encoder
        transform = transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size),interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),                            
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        #print(image_urls)
        start=time.time()
        images=[Image.open(image_url[0].decode()).convert("RGB") for image_url in image_urls]
        print(f"open:{time.time()-start}")
        start=time.time()
        images = [transform(image) for image in images]
        print(f"image preprocess:{time.time()-start}")

        start = torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        start.record()
        images = torch.stack(images).to(device)
        #print(images[0].shape,images.shape)
        #torch.Size([3, 480, 480]) torch.Size([1, 3, 480, 480])
        images_embeds = self.visual_encoder(images)
        images_embeds = images_embeds.numpy(force=True)#to(cpu)
        #print(images_embeds.shape)
        end.record()
        torch.cuda.synchronize()
        print("visual_encoder time:", start.elapsed_time(end)/1000)

        return images_embeds


def blip_vqa_visual_encoder(pretrained="", **kwargs):
    model = BLIP_VQA_VISUAL_ENCODER(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
    #         assert(len(msg.missing_keys)==0)
    return model
