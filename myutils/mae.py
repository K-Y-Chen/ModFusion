import torch

def calculate_mae(pred, target):
    """
    计算分割任务中的 MAE（平均绝对误差）
    
    参数:
    - pred: 模型的预测输出，形状为 (N, H, W) 或 (N, 1, H, W)
    - target: 真实标签，形状为 (N, H, W) 或 (N, 1, H, W)
    
    返回:
    - mae: 平均绝对误差
    """
    # 确保预测和标签形状一致
    if pred.shape != target.shape:
        raise ValueError("预测和真实标签的形状必须一致")
    
    # 计算每个像素的绝对误差
    abs_error = torch.abs(pred - target)
    
    # 计算所有像素的平均值
    mae = torch.mean(abs_error)
    return mae.item()

# # 示例
# # 假设预测值和真实值大小为 (1, 224, 224) 或 (1, 1, 224, 224)
# pred = torch.randn(10, 224, 224)
# target = torch.randn(10, 224, 224)

# # 计算 MAE
# mae = calculate_mae(pred, target)
# print("MAE:", mae)
