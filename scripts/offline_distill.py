import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor


def setup_teacher_model(model_id="google/paligemma2-10b-pt-448", device="cuda:0"):
    print(f"正在載入教師模型: {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print("✅ 教師模型載入完成。")
    return model, processor


def generate_qa_pairs(image_path, model, processor, device="cuda:0"):
    image = Image.open(image_path).convert("RGB")
    conversations = []

    # 蒸餾任務 1: 詳細英文描述 (caption en)
    prompt_caption = "<image>caption en"
    inputs = processor(images=image, text=prompt_caption, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    # 解碼時去除 prefix prompt
    response_caption = processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    conversations.append({
        "instruction": "caption en",
        "response": response_caption
    })

    # 蒸餾任務 2: 物件偵測與空間描述
    prompt_detect = "<image>detect objects; answer question: describe the main elements and spatial relationships"
    inputs = processor(images=image, text=prompt_detect, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    response_detect = processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    conversations.append({
        "instruction": "detect objects; answer question: describe the main elements and spatial relationships",
        "response": response_detect
    })

    return conversations


def run_offline_distillation(
    image_dir="data/raw_images",
    output_json="data/distilled_train.json",
    device="cuda:0"
):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f.lower())[1] in valid_exts
    ]

    if not image_files:
        print(f"❌ 錯誤: 在 {image_dir} 找不到任何圖片！")
        return

    print(f"找到 {len(image_files)} 張圖片，開始進行 10B 知識蒸餾...")
    model, processor = setup_teacher_model(device=device)

    dataset_records = []
    for img_path in tqdm(image_files, desc="蒸餾標註生成中"):
        try:
            convs = generate_qa_pairs(img_path, model, processor, device=device)
            dataset_records.append({
                "image_path": img_path,
                "conversations": convs
            })
        except Exception as e:
            print(f"處理圖片 {img_path} 時發生錯誤: {e}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset_records, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 蒸餾完成！標註已儲存至: {output_json}，共 {len(dataset_records)} 筆樣本。")


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run_offline_distillation(device=device)