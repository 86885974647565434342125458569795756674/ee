from models.blip.med import BertConfig, BertModel, BertLMHeadModel
from models.blip.blip import create_vit, init_tokenizer, load_checkpoint

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch
from torch import nn
import numpy as np
import time


class BLIP_VQA(nn.Module):

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

        self.tokenizer = init_tokenizer()

        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config,
                                      add_pooling_layer=False)

        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)

    def forward(self, image_urls, questions):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Image Preprocesor
#        start = time.time()
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
        image_urls = [image_url.decode("utf-8") for image_url in image_urls]
        images = []
        for image_url in image_urls:
            images.append(transform(Image.open(image_url).convert("RGB")))
        images = torch.stack(images)
        image_batch_size = images.size(0)
#        end = time.time()
#        image_preprocessor_time = end - start

        # Visual Encoder
#        gpu_start=torch.cuda.Event(enable_timing=True)
#        gpu_end=torch.cuda.Event(enable_timing=True)
#        gpu_start.record()
        images = images.to(device)
        images_embeds = self.visual_encoder(images)
#        gpu_end.record()
#        torch.cuda.synchronize()
#        image_encoder_time = gpu_start.elapsed_time(gpu_end)/1000


        # Text Tokenizer
#        start = time.time()
        question_batch_size = questions.shape[0]
        questions = self.tokenizer(
            [question.decode() for question in questions],
            padding="longest",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        )
        questions.input_ids[:, 0] = self.tokenizer.enc_token_id
#        end = time.time()
#        text_tokenizer_time = end - start

        # Text Encoder
#        gpu_start.record()
        images_atts = torch.ones(images_embeds.size()[:-1],
                                 dtype=torch.long).to(device)
        questions = questions.to(device)
        questions_output = self.text_encoder(
            questions.input_ids,
            attention_mask=questions.attention_mask,
            encoder_hidden_states=images_embeds,
            encoder_attention_mask=images_atts,
            return_dict=True,
        )
        num_beams = 1
        questions_states = questions_output.last_hidden_state.repeat_interleave(
            num_beams, dim=0)
#        gpu_end.record()
#        torch.cuda.synchronize()
#        text_encoder_time = gpu_start.elapsed_time(gpu_end)/1000

        # Text Decoder
#        gpu_start.record()
        questions_atts = torch.ones(questions_states.size()[:-1],
                                    dtype=torch.long).to(device)
        model_kwargs = {
            "encoder_hidden_states": questions_states,
            "encoder_attention_mask": questions_atts,
        }
        bos_ids = torch.full(
            (question_batch_size, 1),
            fill_value=self.tokenizer.bos_token_id,
            device=device,
        )
        outputs = self.text_decoder.generate(
            input_ids=bos_ids,
            max_length=10,
            min_length=1,
            num_beams=num_beams,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs,
        )

        answers = [
            self.tokenizer.decode(output, skip_special_tokens=True).encode()
            for output in outputs
        ]
#        gpu_end.record()
#        torch.cuda.synchronize()
#        text_decoder_time = gpu_start.elapsed_time(gpu_end)/1000
        
        '''
        print(f"[image batch size: {image_batch_size}]")
        print("image preprocessor time: ", image_preprocessor_time)
        print("image encoder time: ", image_encoder_time)
        print(f"[question batch size: {question_batch_size}]")
        print("text tokenizer time: ", text_tokenizer_time)
        print("text encoder time: ", text_encoder_time)
        print("text decoder time: ", text_decoder_time)
        print()
        '''

        '''
        with open("/workspace/output.txt", "a") as f:
            print(
                image_preprocessor_time,
                image_encoder_time,
                image_autoscaler_time,
                text_tokenizer_time,
                text_encoder_time,
                text_decoder_time,
                sep="\t",
                file=f,
            )
        '''
        return np.array(answers)


def blip_vqa(pretrained="", **kwargs):
    model = BLIP_VQA(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert len(msg.missing_keys) == 0
    return model
