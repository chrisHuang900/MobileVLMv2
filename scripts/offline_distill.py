import os
import sys
from pathlib import Path

# 將專案根目錄加入模組路徑
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor


class LLaVAInstructDataset(Dataset):
    """
    讀取 llava_instruct_150k.json 並過濾已提取過或不存在的圖片
    """
    def __init__(self, json_path, image_root, output_feat_dir, max_samples=3000):
        print(f"正在載入標註檔: {json_path} ...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples = []
        for item in data:
            if "image" not in item:
                continue

            sample_id = str(item.get("id", len(self.samples)))
            target_feat_path = os.path.join(output_feat_dir, f"{sample_id}.pt")

            # 斷點續傳支援：若已提取過則直接跳過，節省時間
            if os.path.exists(target_feat_path):
                continue

            img_filename = os.path.basename(item["image"])
            img_path = os.path.join(image_root, img_filename)

            if not os.path.exists(img_path):
                continue

            convs = item.get("conversations", [])
            if len(convs) < 2:
                continue

            instruction = convs[0]["value"].replace("<image>\n", "").replace("<image>", "").strip()
            response = convs[1]["value"].strip()

            self.samples.append({
                "id": sample_id,
                "image_path": img_path,
                "instruction": instruction,
                "response": response
            })

            # 達到指定驗證樣本數即停止載入
            if max_samples and len(self.samples) >= max_samples:
                break

        print(f"✅ 本次待處理樣本數: {len(self.samples)} 筆（已自動跳過已快取資料）。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn_factory(processor):
    def collate_fn(batch):
        ids = [b["id"] for b in batch]
        images = [Image.open(b["image_path"]).convert("RGB") for b in batch]
        prompts = [f"<image>{b['instruction']}" for b in batch]
        responses = [b["response"] for b in batch]

        inputs = processor(
            images=images,
            text=prompts,
            suffix=responses,
            return_tensors="pt",
            padding=True
        )
        return ids, inputs
    return collate_fn


def run_offline_feature_distillation(
    image_dir="data/images/train2017",
    json_path="data/llava_instruct_150k.json",
    output_feat_dir="data/teacher_features_10b",
    model_id="google/paligemma2-10b-mix-448",
    batch_size=4,
    max_samples=25000  # 先提取 25,000 筆（約 7.5 ~ 10 小時完成全流程驗證）
):
    os.makedirs(output_feat_dir, exist_ok=True)

    # 1. 建立資料集
    dataset = LLaVAInstructDataset(
        json_path=json_path,
        image_root=image_dir,
        output_feat_dir=output_feat_dir,
        max_samples=max_samples
    )

    if len(dataset) == 0:
        print("所有指定樣本皆已提取完成，或未找到有效資料！")
        return

    # 2. 自動分配雙 GPU 載入模型 (GPU 0 + GPU 1 合併 48GB)
    print(f"正在將 10B 教師模型分配至雙 GPU: {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_factory(processor)
    )

    print(f"開始提取特徵 (目標: {len(dataset)} 筆，Batch Size = {batch_size}) ...")
    
    with torch.inference_mode():
        for batch_ids, batch_inputs in tqdm(dataloader, desc="提取 10B 特徵中"):
            # 將輸入張量傳送到模型輸入層所在的 GPU (model.device)
            pixel_values = batch_inputs["pixel_values"].to(model.device, dtype=torch.bfloat16)
            input_ids = batch_inputs["input_ids"].to(model.device)
            attention_mask = batch_inputs["attention_mask"].to(model.device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            # 取得 10B 語言模型最頂層 Hidden States (形狀: [Batch, Seq_Len, Hidden_Dim])
            last_hidden = outputs.hidden_states[-1]

            # 逐筆儲存至 CPU/硬碟，釋放顯存
            for i, sample_id in enumerate(batch_ids):
                feat_path = os.path.join(output_feat_dir, f"{sample_id}.pt")
                torch.save(last_hidden[i].detach().cpu(), feat_path)

    print(f"\n🎉 驗證特徵提取完成！共儲存 {len(dataset)} 筆特徵至: {output_feat_dir}")
    print("現在你可以直接執行 scripts/train_stage1.py 與 train_stage2.py 驗證整套訓練管線！")


if __name__ == "__main__":
    run_offline_feature_distillation(
        image_dir="data/images/train2017",
        json_path="data/llava_instruct_150k.json",
        output_feat_dir="data/teacher_features_10b",
        model_id="google/paligemma2-10b-mix-448",
        batch_size=4,
        max_samples=25000
    )