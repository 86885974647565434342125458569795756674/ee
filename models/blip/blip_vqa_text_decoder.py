from models.blip.med import BertConfig, BertModel, BertLMHeadModel
from models.blip.blip import create_vit, init_tokenizer, load_checkpoint

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
import time


class BLIP_VQA_TEXT_DECODER(nn.Module):
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

        self.visual_encoder, vision_width = create_vit(
                            vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1
                                    )
        self.tokenizer = init_tokenizer()

        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)

    def forward(self, questions_states,questions_atts):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = questions_states.shape[0]
        #print("batch size:", batch_size)

        # Decoder
        start = torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        start.record()

        num_beams = 1

        questions_states = torch.from_numpy(
            questions_states.reshape(batch_size * num_beams, -1, 768)
        ).to(device)

        questions_atts = torch.from_numpy(questions_atts).to(device)

        model_kwargs = {
            "encoder_hidden_states": questions_states,
            "encoder_attention_mask": questions_atts,
        }
        bos_ids = torch.full(
            (batch_size, 1),
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
            **model_kwargs
        )

        answers = [
            self.tokenizer.decode(output, skip_special_tokens=True).encode()
            for output in outputs
        ]
        end.record()
        torch.cuda.synchronize()
        print("text_decoder time:",  start.elapsed_time(end)/1000)

        return np.array(answers)


def blip_vqa_text_decoder(pretrained="", **kwargs):
    model = BLIP_VQA_TEXT_DECODER(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
    #         assert(len(msg.missing_keys)==0)
    return model
