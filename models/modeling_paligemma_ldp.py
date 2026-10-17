import sys
from pathlib import Path

# 將專案根目錄加入模組搜尋路徑
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from transformers import PaliGemmaForConditionalGeneration
from peft import LoraConfig, get_peft_model
from models.ldp_v2 import LDPv2Projector

def build_paligemma_ldpv2(model_id="google/paligemma2-3b-pt-224", use_lora=True):
    print(f"正在載入基底模型: {model_id} ...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 鎖定核心層級
    core = model.model if hasattr(model, "model") else model
    
    # 1. 凍結視覺編碼器 (SigLIP)
    core.vision_tower.requires_grad_(False)
    print("已成功凍結視覺編碼器 (SigLIP)。")
    
    # 2. 替換 Projector 為 LDP-v2 (確保設為可訓練)
    core.multi_modal_projector = LDPv2Projector(in_dim=1152, out_dim=2048).to(
        dtype=torch.bfloat16, device=model.device
    )
    core.multi_modal_projector.requires_grad_(True)
    print("已替換 Projector 為 LDP-v2 (可訓練)。")
    
    # 3. 對整體模型套用 LoRA (僅鎖定語言模型模組)
    if use_lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        # 直接對最外層 model 套用 LoRA，PEFT 會依 target_modules 自動找尋語言層
        model = get_peft_model(model, lora_config)
        
        # 關鍵補強：PEFT 預設可能會將非 LoRA 層全數凍結，因此重新確保 LDP-v2 維持可訓練
        core_module = model.base_model.model.model if hasattr(model.base_model.model, "model") else model.base_model.model
        core_module.multi_modal_projector.requires_grad_(True)
        print("已成功配置 LoRA 並確保 LDP-v2 處於可訓練狀態。")
    
    return model

if __name__ == "__main__":
    model = build_paligemma_ldpv2()
    print("\n--- 參數量統計 ---")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"總參數量: {total / 1e9:.2f} B")
    print(f"可訓練參數量: {trainable / 1e6:.2f} M ({trainable / total * 100:.2f}%)")
    print("✅ 模型組裝與 LoRA 配置測試完成！")