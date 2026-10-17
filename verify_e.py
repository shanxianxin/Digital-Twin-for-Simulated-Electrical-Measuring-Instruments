import numpy as np
import json

# 加载权重，用纯 numpy 推理
with open("mlp_weights.json") as f:
    layers = json.load(f)


def mlp_predict(x):
    a = np.array(x)
    for i, layer in enumerate(layers):
        W = np.array(layer["W"])
        b = np.array(layer["b"])
        z = W @ a + b
        if i < len(layers) - 1:
            a = np.tanh(z)
        else:
            a = z
    return a[0]


# 复现文档表 1 的测试
def analytical(t, a, w0, z):
    wd = w0 * np.sqrt(1 - z * z)
    return a * (1 - np.exp(-z * w0 * t) * (np.cos(wd * t) + (z * w0 / wd) * np.sin(wd * t)))


w0, z, a_t, dt = 4.2, 0.7, 68.0, 1 / 60
alpha, nu = 0.0, 0.0

for k in range(8):
    t = k * dt
    # 欧拉
    acc = w0 * w0 * (a_t - alpha) - 2 * z * w0 * nu
    nu_next = nu + dt * acc
    alpha_num = alpha + dt * nu_next
    # 解析
    alpha_ana = analytical(t, a_t, w0, z)
    # MLP 残差
    feat = [t * w0, a_t / 160, alpha_num / 160, nu_next / (w0 * 160), w0 / 10, z, dt * w0]
    res = mlp_predict(feat) * 160.0
    alpha_hyb = alpha_num + res

    print(f"t={t:.3f} | ana={alpha_ana:7.3f} | num={alpha_num:7.3f} "
          f"| err_num={alpha_ana - alpha_num:+.4f} | hyb={alpha_hyb:7.3f} "
          f"| err_hyb={alpha_ana - alpha_hyb:+.4f}")

    alpha, nu = alpha_num, nu_next