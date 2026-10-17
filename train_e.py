import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import json

# 固定随机种子
np.random.seed(42)
torch.manual_seed(42)

# 1. 加载数据 & 轨迹元信息（每条轨迹真实长度）
X_np = np.load("X.npy")
Y_np = np.load("Y.npy").reshape(-1, 1)

with open("traj_meta.json", "r") as f:
    traj_meta = json.load(f)

n_traj_total = len(traj_meta)
print(f"总轨迹数量 {n_traj_total}")

# 2. 8:2 划分训练/验证轨迹
train_ratio = 0.8
n_traj_train = int(n_traj_total * train_ratio)
n_traj_val = n_traj_total - n_traj_train

traj_idx = torch.randperm(n_traj_total)
train_traj_idx = traj_idx[:n_traj_train]
val_traj_idx = traj_idx[n_traj_train:]
print(f"训练轨迹 {n_traj_train} | 验证轨迹 {n_traj_val}")

# 3. 根据 meta 精准截取每条轨迹（核心修复：不等长轨迹正确切分）
def get_traj_samples(traj_indices, meta, X_all, Y_all):
    xs, ys = [], []
    ptr = 0
    for idx in range(len(meta)):
        length = meta[idx]["length"]
        # 当前轨迹是否属于选中的轨迹索引
        if idx in traj_indices:
            xs.append(X_all[ptr:ptr+length])
            ys.append(Y_all[ptr:ptr+length])
        ptr += length
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

X_train_np, Y_train_np = get_traj_samples(train_traj_idx.tolist(), traj_meta, X_np, Y_np)
X_val_np, Y_val_np = get_traj_samples(val_traj_idx.tolist(), traj_meta, X_np, Y_np)

# 4. 转 tensor
X_train = torch.tensor(X_train_np, dtype=torch.float32)
Y_train = torch.tensor(Y_train_np, dtype=torch.float32)
X_val = torch.tensor(X_val_np, dtype=torch.float32)
Y_val = torch.tensor(Y_val_np, dtype=torch.float32)

train_ds = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)

# 5. MLP 模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.6)
loss_fn = nn.MSELoss()

best_val = float("inf")

# 6. 训练
for epoch in range(3000):
    model.train()
    total_train_loss = 0.0
    for bx, by in train_loader:
        optimizer.zero_grad()
        pred = model(bx)
        loss = loss_fn(pred, by)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * bx.shape[0]
    scheduler.step()
    avg_train_loss = total_train_loss / len(train_ds)

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val)
            val_loss = loss_fn(y_pred_val, Y_val).item()
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pt")
        print(f"Epoch {epoch:4d} | train {avg_train_loss:.3e} | val {val_loss:.3e}")

# 加载最优模型
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# 7. 计算角度 RMSE（和你实验B完全对齐）
with torch.no_grad():
    y_pred_val = model(X_val)

res_true_norm = Y_val_np
res_pred_norm = y_pred_val.detach().numpy()

res_true_deg = res_true_norm * 160.0
res_pred_deg = res_pred_norm * 160.0

rmse_euler = np.sqrt(np.mean(np.square(res_true_deg)))
rmse_corrected = np.sqrt(np.mean(np.square(res_true_deg - res_pred_deg)))
print(f"\n验证集【原生半隐式欧拉RMSE】={rmse_euler:.4f} °")
print(f"验证集【MLP校正后RMSE】={rmse_corrected:.4f} °")

# 导出权重
def export_mlp(model):
    layers = []
    for m in model.net:
        if isinstance(m, nn.Linear):
            layers.append({
                "W": m.weight.detach().numpy().tolist(),
                "b": m.bias.detach().numpy().tolist(),
            })
    return layers

with open("mlp_weights.json", "w") as f:
    json.dump(export_mlp(model), f)
print("权重导出 mlp_weights.json 完成")