from models.blip.med import BertConfig, BertModel, BertLMHeadModel
from models.blip.blip import create_vit, init_tokenizer, load_checkpoint

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
import time


class BLIP_VQA_TEXT_ENCODER(nn.Module):
    def __init__(
        self,
        med_config="configs/med_config.json",
        image_size=480,
        vit="large",
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

        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1
        )

        self.tokenizer = init_tokenizer()

        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward_time(self, images_embeds, questions):

        start= time.perf_counter()
        batch_size = len(questions)
        questions = self.tokenizer(
            [question.decode("utf-8") for question in questions],
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        )   
        #print(questions)
        print("text_preprocess time:",  time.perf_counter()-start)
        
        start = torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        start.record()
        questions.to(self.device)
        images_atts = torch.ones(images_embeds.shape[:-1], dtype=torch.long).to(self.device)

        questions.input_ids[:, 0] = self.tokenizer.enc_token_id
        questions_output = self.text_encoder(
            questions.input_ids,
            attention_mask=questions.attention_mask,
            encoder_hidden_states=images_embeds,
            encoder_attention_mask=images_atts,
            return_dict=True,
        )
        num_beams = 1
        questions_states =   questions_output.last_hidden_state.repeat_interleave(num_beams, dim=0)#.reshape(batch_size, num_beams, -1, 768)

        end.record()
        torch.cuda.synchronize()
        print("text_encoder time:", start.elapsed_time(end)/1000)

        return questions_states

    def forward(self, images_embeds, questions):
        batch_size = len(questions)
        #print("batch size:", batch_size)

        # Text Encoder
        
        questions = self.tokenizer(
            [question.decode("utf-8") for question in questions],
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)

#        images_embeds = torch.from_numpy(images_embeds).to(self.device)
        images_atts = torch.ones(images_embeds.shape[:-1], dtype=torch.long).to(self.device)

        questions.input_ids[:, 0] = self.tokenizer.enc_token_id
        questions_output = self.text_encoder(
            questions.input_ids,
            attention_mask=questions.attention_mask,
            encoder_hidden_states=images_embeds,
            encoder_attention_mask=images_atts,
            return_dict=True,
        )
        num_beams = 1
        questions_states =   questions_output.last_hidden_state.repeat_interleave(num_beams, dim=0)#.reshape(batch_size, num_beams, -1, 768)
            #.numpy(force=True)

        return questions_states


def blip_vqa_text_encoder(pretrained="", **kwargs):
    model = BLIP_VQA_TEXT_ENCODER(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
    #         assert(len(msg.missing_keys)==0)
    model.to(model.device)
    return model
