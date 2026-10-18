import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class DistilledVLMDataset(Dataset):
    def __init__(self, json_file, processor, max_length=256, num_image_tokens=64):
        self.processor = processor
        self.max_length = max_length
        self.num_image_tokens = num_image_tokens
        
        with open(json_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        self.samples = []
        for item in raw_data:
            img_path = item["image_path"]
            for conv in item.get("conversations", []):
                self.samples.append({
                    "image_path": img_path,
                    "instruction": conv["instruction"],
                    "response": conv["response"]
                })
        print(f"成功載入資料集，共 {len(self.samples)} 筆訓練問答。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        
        # 關鍵修正：LDP-v2 輸出 64 個 token，因此 prompt 開頭必須剛好對齊 64 個 <image> token
        image_tokens = "<image>" * self.num_image_tokens
        prompt_text = f"{image_tokens}{sample['instruction']}\n"
        full_text = f"{prompt_text}{sample['response']}"
        
        # 1. 處理影像 (固定解析度 224x224)
        pixel_values = self.processor.image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        # 2. 處理文字 Token (不交給 processor 自動擴展 image token，手動精確 tokenize)
        text_inputs = self.processor.tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        prompt_inputs = self.processor.tokenizer(
            prompt_text,
            return_tensors="pt"
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]
        
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        
        # 3. 建立 Labels (-100 忽略 prompt 與 padding)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels
        }