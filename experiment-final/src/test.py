import torch
import torch.profiler

# 确保张量在 GPU 上，使用 float16 精度
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)

# 配置 TF32 选项（可选，根据需求）
torch.backends.cuda.matmul.allow_tf32 = True

# 使用 PyTorch Profiler 记录操作
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CUDA,  # 捕获 CUDA 调用
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),  # 记录到文件
    record_shapes=True,
    with_stack=True,
) as prof:
    result = torch.mm(a, b)

# 打印概要信息
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
