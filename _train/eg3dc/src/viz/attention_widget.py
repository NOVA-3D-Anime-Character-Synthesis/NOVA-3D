import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设的注意力权重矩阵
# 这里我们随机生成数据作为示例
# 真实情况下，这将是模型的输出
# attention_weights = torch.rand(B * H * W * D, 1, num_heads, num_heads)
def attention_draw(attention_weights):
    # 选择一个样本和注意力头进行可视化
    # 这里我们简单地取平均值 across all heads
    B = 4  # 批量大小
    H = W = 96  # 高度、宽度、深度
    D = 48
    num_heads = 8
    # attention_weights = torch.rand(B * H * W * D, 1, num_heads, num_heads)
    attention_weights = attention_weights.clone().detach().cpu()
    sample_attention = attention_weights.view(-1, H, W, D, num_heads).mean(dim=(-1,))[0] # [96,96,48]

    # 将注意力权重转换为numpy数组
    attention_weights_np = sample_attention.numpy()

    # 创建一个三维网格来可视化热图
    x, y, z = np.meshgrid(np.arange(H), np.arange(W), np.arange(D))

    # 创建绘图
    fig = plt.figure(figsize=(512, 512))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制每个点，颜色根据注意力权重来确定
    # 这里我们使用归一化的权重来映射到颜色映射
    norm = plt.Normalize(attention_weights_np.min(), attention_weights_np.max())
    colors = plt.cm.viridis(norm(attention_weights_np))

    # 对于每个点，绘制一个小方块（体素）
    for i in range(H):
        for j in range(W):
            for k in range(D):
                ax.scatter(x[i, j, k], y[i, j, k], z[i, j, k], color=colors[i, j, k])

    # 设置图表标题和坐标轴标签
    ax.set_title("3D Attention Weight Visualization")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    ax.set_zlabel("Depth")

    # 显示图表
    plt.show()
    plt.savefig('./temp/attention_plot.png')