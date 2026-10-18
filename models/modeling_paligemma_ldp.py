import sys
from pathlib import Path

# 將專案根目錄加入模組搜尋路徑
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from transformers import PaliGemmaForConditionalGeneration
from peft import LoraConfig, get_peft_model
from models.ldp_v2 import LDPv2Projector


class PaliGemmaLDPv2ForConditionalGeneration(nn.Module):
    """
    包裝 PaliGemma 2，並以直接特徵拼接取代原廠脆弱的 masked_scatter
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def __getattr__(self, name):
        """
        屬性穿透代理：當外部存取不存在於包裝類別的屬性時，
        自動轉發至底層模型或其核心模組。
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            core = self.base_model.model if hasattr(self.base_model, "model") else self.base_model
            if hasattr(core, name):
                return getattr(core, name)
            return getattr(self.base_model, name)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None, **kwargs):
        core = self.base_model.model if hasattr(self.base_model, "model") else self.base_model

        # 1. 取得文字 Embedding
        embed_tokens = core.language_model.get_input_embeddings()
        text_embeds = embed_tokens(input_ids)  # 形狀: (B, Text_Len, 2304)

        # 2. 取得影像特徵並通過 LDP-v2
        if pixel_values is not None:
            # SigLIP 抽取特徵: (B, 256, 1152)
            image_outputs = core.vision_tower(pixel_values)
            selected_image_feature = image_outputs.last_hidden_state

            # LDP-v2 降採樣: (B, 64, 2304)
            image_features = core.multi_modal_projector(selected_image_feature)

            # 3. 拼接影像與文字特徵: (B, 64 + Text_Len, 2304)
            inputs_embeds = torch.cat([image_features, text_embeds], dim=1)

            # 4. 同步擴展 attention_mask
            if attention_mask is not None:
                img_mask = torch.ones(
                    (attention_mask.shape[0], image_features.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                attention_mask = torch.cat([img_mask, attention_mask], dim=1)

            # 5. 同步擴展 labels (影像區域不計 Loss，設為 -100)
            if labels is not None:
                img_labels = torch.full(
                    (labels.shape[0], image_features.shape[1]),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                labels = torch.cat([img_labels, labels], dim=1)
        else:
            inputs_embeds = text_embeds

        # 6. 送入語言模型計算
        outputs = core.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        return outputs


def build_paligemma_ldpv2(model_id="google/paligemma2-3b-pt-224", use_lora=True):
    print(f"正在載入基底模型: {model_id} ...")
    raw_model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    core = raw_model.model if hasattr(raw_model, "model") else raw_model

    # 1. 凍結視覺編碼器 (SigLIP)
    core.vision_tower.requires_grad_(False)
    print("已成功凍結視覺編碼器 (SigLIP)。")

    # 2. 自動偵測語言模型隱藏層維度 (Gemma 2-2B 為 2304)
    llm_hidden_size = core.language_model.config.hidden_size
    print(f"偵測到語言模型隱藏層維度: {llm_hidden_size}")

    # 3. 換裝 LDP-v2 Projector (可訓練)
    core.multi_modal_projector = LDPv2Projector(in_dim=1152, out_dim=llm_hidden_size).to(
        dtype=torch.bfloat16, device=raw_model.device
    )
    core.multi_modal_projector.requires_grad_(True)
    print(f"已替換 Projector 為 LDP-v2 (1152 -> {llm_hidden_size}，可訓練)。")

    # 4. 語言模型掛載 LoRA
    if use_lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        core.language_model = get_peft_model(core.language_model, lora_config)
        print("已成功在語言模型 (Gemma 2) 上掛載 LoRA。")

    # 5. 包裝為自定義直接拼接前向模型
    model = PaliGemmaLDPv2ForConditionalGeneration(raw_model)
    return model


if __name__ == "__main__":
    model = build_paligemma_ldpv2()
    print("\n--- 參數量統計 ---")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"總參數量: {total / 1e9:.2f} B")
    print(f"可訓練參數量: {trainable / 1e6:.2f} M ({trainable / total * 100:.2f}%)")
    print("✅ 模型組裝測試完成！")