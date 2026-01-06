# models/encoder.py
import torch.nn as nn
from transformers import CLIPModel


class MWPEncoder(nn.Module):
    def __init__(self, clip_path, device="cuda"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_path)

        self.image_cls_fc = nn.Linear(512, 96)
        self.image_tokens_fc = nn.Linear(768, 96)

        self.device = device
        self.to(device)

    def forward(self,
                caption_input_ids,
                caption_attention_mask,
                pixel_values,
                image_description_input_ids=None,
                image_description_attention_mask=None):

        out = self.clip(
            input_ids=caption_input_ids,
            attention_mask=caption_attention_mask,
            pixel_values=pixel_values
        )

        cap_emb = out.text_embeds
        img_emb = self.image_cls_fc(out.image_embeds)

        cap_tok = out.text_model_output[0]
        img_tok = self.image_tokens_fc(out.vision_model_output[0])

        if image_description_input_ids is None:
            return cap_emb, img_emb, cap_tok, img_tok, None, None

        desc_out = self.clip(
            input_ids=image_description_input_ids,
            attention_mask=image_description_attention_mask
        )

        return (
            cap_emb, img_emb, cap_tok, img_tok,
            desc_out.text_embeds,
            desc_out.text_model_output[0]
        )
