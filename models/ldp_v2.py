import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LDPv2Projector(nn.Module):
    """
    自適應多解析度 LDP-v2 輕量化投影層：
    - 支援 224x224 (256 tokens -> 64 tokens)
    - 支援 448x448 (1024 tokens -> 256 tokens)
    將 SigLIP 影像特徵進行空間 2x2 (4倍) 降採樣，並映射至 3B LLM 隱藏維度 (預設 2048)。
    """
    def __init__(self, in_dim=1152, out_dim=2048):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 卷積主幹分支 (深度可分離卷積，stride=2 進行空間降採樣)
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

        # 池化殘差分支 (1x1 卷積變換維度)
        self.pool_proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

        self.act = nn.GELU()

    def forward(self, x):
        # x 形狀: (B, N, C)，例如 224 解析度下為 (B, 256, 1152)
        B, N, C = x.shape
        H = W = int(math.isqrt(N))
        assert H * W == N, f"輸入 token 數 {N} 無法開整數根號還原為 2D 特徵圖！"

        # 轉置為 2D 空間特徵: (B, C, H, W)
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)

        # 卷積降採樣 (H, W 變為 H//2, W//2)
        out_conv = self.conv_branch(x_2d)
        target_h, target_w = out_conv.shape[2], out_conv.shape[3]

        # 自適應池化分支：尺寸動態對齊卷積輸出
        out_pool = F.adaptive_avg_pool2d(x_2d, (target_h, target_w))
        out_pool = self.pool_proj(out_pool)

        # 融合兩路特徵並通過激活函數
        out_2d = self.act(out_conv + out_pool)

        # 展平回序列形式: (B, target_N, out_dim) -> (B, 64, 2048)
        out = out_2d.flatten(2).transpose(1, 2)
        return out


if __name__ == "__main__":
    # 單元測試: 224x224 解析度 (256 tokens -> 64 tokens，對應 3B 規格)
    print("--- 測試 224 解析度 (3B 規格) ---")
    x_224 = torch.randn(2, 256, 1152)
    proj_3b = LDPv2Projector(in_dim=1152, out_dim=2048)
    out_224 = proj_3b(x_224)
    print(f"輸入特徵形狀: {x_224.shape}")
    print(f"LDP-v2 輸出形狀: {out_224.shape}")
    assert out_224.shape == (2, 64, 2048), "224 規格 3B 維度計算錯誤！"
    print("224 規格 3B 測試通過！")