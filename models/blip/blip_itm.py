import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from .med import BertConfig, BertModel
from .blip import create_vit, init_tokenizer, load_checkpoint


class BLIP_ITM(nn.Module):

    def __init__(
        self,
        med_config='configs/med_config.json',
        image_size=384,
        vit='base',
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        embed_dim=256,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.image_size = image_size
        self.visual_encoder, vision_width = create_vit(vit, image_size,
                                                       vit_grad_ckpt,
                                                       vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config,
                                      add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

    def forward(self, image_urls, captions):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Preprocess
        image_urls = [image_url.decode() for image_url in image_urls]
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
        for image_url in image_urls:
            images.append(transform(Image.open(image_url).convert("RGB")))
        images = torch.stack(images).to(device)

        image_embeds = self.visual_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(device)

        text = self.tokenizer([caption.decode() for caption in captions],
                              padding='max_length',
                              truncation=True,
                              max_length=35,
                              return_tensors="pt").to(device)

        output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        itm_output = self.itm_head(output.last_hidden_state[:, 0, :])
        score = torch.nn.functional.softmax(itm_output, dim=1)[:, 1]
        return score.cpu().numpy()


def blip_itm(pretrained='', **kwargs):
    model = BLIP_ITM(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert (len(msg.missing_keys) == 0)
    return model
