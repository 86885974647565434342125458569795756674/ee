from .med import BertConfig
from .nlvr_encoder import BertModel
from .vit import interpolate_pos_embed
from .blip import create_vit, init_tokenizer, is_url

from timm.models.hub import download_cached_file

import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class BLIP_NLVR_VISUAL_ENCODER(nn.Module):

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

        self.image_size = image_size
        self.visual_encoder, vision_width = create_vit(vit,
                                                       image_size,
                                                       vit_grad_ckpt,
                                                       vit_ckpt_layer,
                                                       drop_path_rate=0.1)
        # self.tokenizer = init_tokenizer()
        # med_config = BertConfig.from_json_file(med_config)
        # med_config.encoder_width = vision_width
        # self.text_encoder = BertModel(config=med_config,
        #                               add_pooling_layer=False)

        # self.cls_head = nn.Sequential(
        #     nn.Linear(
        #         self.text_encoder.config.hidden_size,
        #         self.text_encoder.config.hidden_size,
        #     ),
        #     nn.ReLU(),
        #     nn.Linear(self.text_encoder.config.hidden_size, 2),
        # )

    def forward(self, image0_urls, image1_urls):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#        start = torch.cuda.Event(enable_timing=True)
#        end=torch.cuda.Event(enable_timing=True)
#        start.record()

        # Preprocess
        image0_urls = [image0_url.decode() for image0_url in image0_urls]
        image1_urls = [image1_url.decode() for image1_url in image1_urls]
        transform = transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        images = []
        for image0_url in image0_urls:
            images.append(transform(Image.open(image0_url).convert("RGB")))
        for image1_url in image1_urls:
            images.append(transform(Image.open(image1_url).convert("RGB")))
        images = torch.stack(images)

        # Visual Encoder
        images = images.to(device)
        images_embeds = self.visual_encoder(images).numpy(force=True)
        # images_atts = torch.ones(images_embeds.size()[:-1],
        #                          dtype=torch.long).to(device)

#        end.record()
#        torch.cuda.synchronize()
#        print("visual_encoder time:", start.elapsed_time(end)/1000)

        return images_embeds


def blip_nlvr_visual_encoder(pretrained="", **kwargs):
    model = BLIP_NLVR_VISUAL_ENCODER(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        # print("missing keys:")
        # print(msg.missing_keys)
    return model


def load_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename,
                                           check_hash=False,
                                           progress=True)
        checkpoint = torch.load(cached_file, map_location="cpu")
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location="cpu")
    else:
        raise RuntimeError("checkpoint url or path is invalid")
    state_dict = checkpoint["model"]

    state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
        state_dict["visual_encoder.pos_embed"], model.visual_encoder)

    for key in list(state_dict.keys()):
        if "crossattention.self." in key:
            new_key0 = key.replace("self", "self0")
            new_key1 = key.replace("self", "self1")
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]
        elif "crossattention.output.dense." in key:
            new_key0 = key.replace("dense", "dense0")
            new_key1 = key.replace("dense", "dense1")
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print("load checkpoint from %s" % url_or_filename)
    return model, msg
