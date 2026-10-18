import sys
from pathlib import Path

# 將專案根目錄加入路徑
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from models.modeling_paligemma_ldp import build_paligemma_ldpv2
from utils.dataset import DistilledVLMDataset
from tqdm import tqdm

def train_stage1():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用裝置: {device}")
    
    # 1. 載入模型 (Stage 1 不掛載 LoRA)
    model = build_paligemma_ldpv2(model_id="google/paligemma2-3b-pt-224", use_lora=False)
    processor = AutoProcessor.from_pretrained("google/paligemma2-3b-pt-224")
    
    # 取得底層核心
    raw = model.base_model
    core = raw.model if hasattr(raw, "model") else raw
    
    # 鎖死語言模型，只開放 Projector 訓練
    core.language_model.requires_grad_(False)
    core.multi_modal_projector.requires_grad_(True)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Stage 1 可訓練參數量: {sum(p.numel() for p in trainable_params) / 1e6:.2f} M")
    
    # 2. 準備資料
    dataset = DistilledVLMDataset("data/distilled_train.json", processor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 3. 優化器與超參數
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.01)
    model.train()
    
    # 4. 跑 1 個 Epoch 測試前向與反向傳播
    epochs = 1
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16 if device=="cuda" else torch.float32)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
    # 5. 儲存 Stage 1 訓練好的 LDP-v2 權重
    Path("checkpoints").mkdir(exist_ok=True)
    save_path = "checkpoints/ldp_v2_stage1.pth"
    torch.save(core.multi_modal_projector.state_dict(), save_path)
    print(f"✅ Stage 1 訓練測試完成，LDP-v2 權重已儲存至 {save_path}")

if __name__ == "__main__":
    train_stage1()  