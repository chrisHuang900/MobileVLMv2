import torch
import torch.nn as nn
import math

class LDPv2Projector(nn.Module):
    """
    LDP-v2 (Lightweight Downsample Projector v2)
    將 SigLIP 輸出的 (B, 256, 1152) 降採樣壓縮成 (B, 64, 2048)
    Token 數量減少 75%，大幅降低語言模型的注意力計算負擔
    """
    def __init__(self, in_dim=1152, out_dim=2048):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 1. 主分支：Depthwise Separable Convolution (長寬減半: 16x16 -> 8x8)
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),              # 跨通道特徵融合
            nn.GELU(),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim), # 深度卷積 (空間降採樣)
            nn.GELU(),
            nn.Conv2d(in_dim, out_dim, kernel_size=1),             # 投影至語言模型維度
            nn.GELU()
        )
        
        # 2. 殘差分支：Adaptive Avg Pooling (固定下採樣為 8x8) + 1x1 投影
        self.pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.GELU()
        )
        
        # 3. 輸出層 LayerNorm
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x 輸入形狀: (Batch, Tokens, Dim) = (B, 256, 1152)
        B, N, C = x.shape
        H = W = int(math.sqrt(N))  # 16 x 16
        
        # 轉成 2D 空間特徵圖: (B, Dim, H, W)
        x_2d = x.transpose(1, 2).view(B, C, H, W)
        
        # 雙流並行計算
        out_conv = self.conv_branch(x_2d)  # (B, out_dim, 8, 8)
        out_pool = self.pool_branch(x_2d)  # (B, out_dim, 8, 8)
        
        # 雙流相加融合
        out_2d = out_conv + out_pool       # (B, out_dim, 8, 8)
        
        # 展平回序列特徵: (B, 64, out_dim)
        out = out_2d.flatten(2).transpose(1, 2)
        out = self.norm(out)
        return out

if __name__ == "__main__":
    # 單元測試：確認維度轉換
    test_input = torch.randn(2, 256, 1152)
    projector = LDPv2Projector(1152, 2048)
    test_output = projector(test_input)
    print(f"輸入特徵形狀: {test_input.shape}")
    print(f"LDP-v2 降採樣後特徵形狀: {test_output.shape}")
    assert test_output.shape == (2, 64, 2048), "維度轉換錯誤！"
    print("✅ LDP-v2 單元測試通過！")