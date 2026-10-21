import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor
from tqdm import tqdm
from models.modeling_paligemma_ldp import build_paligemma_ldpv2


class DistilledDataset(Dataset):
    def __init__(self, json_file, processor):
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = processor
        self.samples = []
        for item in self.data:
            img_path = item["image_path"]
            for conv in item["conversations"]:
                self.samples.append({
                    "image_path": img_path,
                    "prompt": conv["instruction"],
                    "suffix": conv["response"]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        prompt = sample["prompt"]
        suffix = sample["suffix"]

        if not prompt.startswith("<image>"):
            prompt = "<image>" + prompt

        inputs = self.processor(
            images=image,
            text=prompt,
            suffix=suffix,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": inputs["labels"].squeeze(0)
        }


def train_stage2():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    # 1. 建立 3B-224 模型並掛載 LoRA
    model_id = "google/paligemma2-3b-pt-224"
    print(f"正在組裝 Stage 2 模型 ({model_id} 開啟 LoRA) ...")
    model = build_paligemma_ldpv2(
        model_id=model_id,
        use_lora=True,
        device=device
    )

    # 2. 載入 Stage 1 訓練好的 3B 規格 LDP-v2 權重 (維度 2048)
    stage1_ckpt = "checkpoints/ldp_v2_stage1_3b_224.pth"
    core = model.base_model.model if hasattr(model.base_model, "model") else model.base_model
    if os.path.exists(stage1_ckpt):
        print(f"正在載入 Stage 1 Projector 權重: {stage1_ckpt} ...")
        core.multi_modal_projector.load_state_dict(torch.load(stage1_ckpt, map_location=device, weights_only=True))
        print("✅ Stage 1 權重載入成功！")
    else:
        print(f"⚠️ 找不到 {stage1_ckpt}，將從隨機權重直接訓練。")

    model.train()
    model.to(device)

    # 3. 檢查可訓練參數量 (LDP-v2 + LoRA)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_num = sum(p.numel() for p in trainable_params)
    print(f"Stage 2 可訓練參數量 (LDP-v2 + LoRA): {trainable_num / 1e6:.2f} M")

    # 4. 載入 224 Processor 與資料
    processor = AutoProcessor.from_pretrained(model_id)
    dataset = DistilledDataset("data/distilled_train.json", processor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"成功載入資料集，共 {len(dataset)} 筆訓練樣本。")

    # 5. 優化器設定
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-5, weight_decay=0.01)

    # 6. 訓練迴圈
    num_epochs = 2
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Stage 2 Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 7. 儲存 Stage 2 產出權重
    save_path = "checkpoints/stage2_3b_224_final"
    os.makedirs(save_path, exist_ok=True)
    torch.save(core.multi_modal_projector.state_dict(), f"{save_path}/ldp_v2_final.pth")
    if hasattr(core.language_model, "save_pretrained"):
        core.language_model.save_pretrained(f"{save_path}/lora_adapter")
    print(f"\n✅ Stage 2 (3B-224) 訓練完成！權重已儲存至 {save_path}/")


if __name__ == "__main__":
    train_stage2()