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


def train_stage1():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    # 1. 建立模型 (Stage 1 不開 LoRA)
    print("正在組裝 Stage 1 模型 (僅訓練 LDP-v2 Projector) ...")
    model = build_paligemma_ldpv2(
        model_id="google/paligemma2-10b-pt-448",
        use_lora=False,
        device=device
    )

    core = model.base_model.model if hasattr(model.base_model, "model") else model.base_model
    # 確保只有 Projector 可訓練
    core.language_model.requires_grad_(False)
    core.multi_modal_projector.requires_grad_(True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_num = sum(p.numel() for p in trainable_params)
    print(f"Stage 1 可訓練參數量: {trainable_num / 1e6:.2f} M")

    # 2. 載入資料
    processor = AutoProcessor.from_pretrained("google/paligemma2-10b-pt-448")
    dataset = DistilledDataset("data/distilled_train.json", processor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"成功載入資料集，共 {len(dataset)} 筆訓練問答。")

    # 3. 優化器設定
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.01)

    model.train()
    num_epochs = 1
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Stage 1 Epoch {epoch + 1}/{num_epochs}")
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

    # 4. 儲存 Stage 1 權重
    save_ckpt = "checkpoints/ldp_v2_stage1.pth"
    torch.save(core.multi_modal_projector.state_dict(), save_ckpt)
    print(f"\n✅ Stage 1 訓練測試完成，LDP-v2 權重已儲存至 {save_ckpt}")


if __name__ == "__main__":
    train_stage1()