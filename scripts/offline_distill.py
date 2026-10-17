import os
import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

def setup_teacher_model(model_id="google/paligemma2-3b-pt-224", device="cuda"):
    """
    載入 10B 教師模型。 (10b-pt-448)
    若在單張 24GB 顯卡上執行，建議使用 bfloat16。
    若顯存吃緊，可加入 load_in_8bit=True 或 load_in_4bit=True。
    """
    print(f"正在載入教師模型: {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    return model, processor

def generate_qa_pairs(image_path, model, processor, device="cuda"):
    """
    引導教師模型針對給定影像生成高密度的問答對
    """
    image = Image.open(image_path).convert("RGB")
    
    # 定義提示詞引導教師生成細節描述與推理問答
    prompts = [
        "caption en",  # 詳細英文場景描述
        "detect objects; answer question: describe the main elements and spatial relationships in this image"
    ]
    
    qa_results = []
    
    for prompt in prompts:
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        # 輸入特徵轉為與模型相符的 bfloat16
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False  # 使用貪婪搜尋維持生成確定性與穩定度
            )
        
        # 解碼輸出並去除 prompt 前綴
        input_len = inputs["input_ids"].shape[1]
        decoded = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip()
        
        qa_results.append({
            "instruction": prompt,
            "response": decoded
        })
        
    return qa_results

def run_distillation(image_dir="data/raw_images", output_json="data/distilled_train.json"):
    Path("data").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, processor = setup_teacher_model(device=device)
    
    image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    if not image_paths:
        print(f"警告：在 {image_dir} 中未找到圖片，請放入少許圖片進行測試！")
        return
        
    print(f"共找到 {len(image_paths)} 張圖片，開始進行離線生成...")
    dataset = []
    
    for img_path in tqdm(image_paths):
        try:
            qa_pairs = generate_qa_pairs(img_path, model, processor, device=device)
            dataset.append({
                "image_path": str(img_path),
                "conversations": qa_pairs
            })
        except Exception as e:
            print(f"處理 {img_path} 時發生錯誤: {e}")
            continue
            
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"蒸餾資料已成功輸出至: {output_json} (共 {len(dataset)} 筆)")

if __name__ == "__main__":
    run_distillation()