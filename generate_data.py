import numpy as np
import json


# ★ 新增：带初始条件的解析解
def analytical_ic(t, target, omega0, zeta, alpha0, nu0):
    wd = omega0 * np.sqrt(1 - zeta ** 2)
    A = alpha0 - target
    B = (nu0 + zeta * omega0 * A) / wd
    return target + np.exp(-zeta * omega0 * t) * (
            A * np.cos(wd * t) + B * np.sin(wd * t)
    )


def euler_run(alpha_target, omega0, zeta, dt, n_steps, alpha0, nu0):
    # ★ 不再写死，使用传入的初始状态
    alpha = alpha0
    nu = nu0
    alphas = [alpha]
    nus = [nu]
    for _ in range(n_steps):
        acc = omega0 ** 2 * (alpha_target - alpha) - 2 * zeta * omega0 * nu
        nu = nu + dt * acc
        alpha = alpha + dt * nu
        alphas.append(alpha)
        nus.append(nu)
    return np.array(alphas), np.array(nus)


def build_dataset(n_traj=2000, T_total=2.0):
    X_list, Y_list = [], []
    traj_meta = []

    DT_CHOICES = [1 / 60, 1 / 30, 1 / 20, 1 / 15, 1 / 10]

    for _ in range(n_traj):
        omega0 = np.random.uniform(2.0, 8.0)
        zeta = np.random.uniform(0.3, 0.98)
        alpha_target = np.random.uniform(-160.0, 160.0)

        # ★ 关键修改：随机化初始角度和初始角速度
        # 模拟真实场景：从任意稳态位置（-160到160）起步，角速度默认为0
        # 30% 小阶跃 + 70% 大阶跃
        if np.random.rand() < 0.3:
            alpha0 = alpha_target + np.random.uniform(-5.0, 5.0)
        else:
            alpha0 = np.random.uniform(-160.0, 160.0)

        # 保证 alpha0 在合法范围内（30% 分支可能越界）
        alpha0 = float(np.clip(alpha0, -160.0, 160.0))
        nu0 = 0.0

        dt = np.random.choice(DT_CHOICES)

        n_steps = int(round(T_total / dt))
        alphas, nus = euler_run(alpha_target, omega0, zeta, dt, n_steps, alpha0, nu0)

        traj_len = n_steps + 1
        traj_meta.append({
            "length": traj_len,
            "dt": dt,
            "omega0": float(omega0),
            "zeta": float(zeta),
            "alpha_target": float(alpha_target),
            "alpha0": float(alpha0),  # ★ 建议存入 meta
            "nu0": float(nu0)  # ★ 建议存入 meta
        })

        for n in range(traj_len):
            t = n * dt
            alpha_num = alphas[n]
            nu_num = nus[n]
            # ★ 使用带初始条件的解析解
            alpha_ana = analytical_ic(t, alpha_target, omega0, zeta, alpha0, nu0)
            residual = alpha_ana - alpha_num

            # ★ 特征工程优化：将绝对角度转为相对初始角度的偏移，消除绝对位置的影响
            # 原来的 feat 也可以保留，但相对特征能极大加速网络收敛
            feat = [
                t * omega0,
                (alpha_target - alpha0) / 160.0,  # 改为相对目标
                (alpha_num - alpha0) / 160.0,  # 改为相对当前
                nu_num / (omega0 * 160.0),
                omega0 / 10.0,
                zeta,
                dt * omega0,
                alpha0 / 160.0  # ★ 把初始角度本身作为一个特征传入
            ]
            X_list.append(feat)
            Y_list.append(residual / 160.0)

    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32), traj_meta

if __name__ == "__main__":
    np.random.seed(42)
    X, Y, traj_meta = build_dataset(n_traj=2000)
    np.save("X.npy", X)
    np.save("Y.npy", Y)
    with open("traj_meta.json", "w") as f:
        json.dump(traj_meta, f)


    print(f" X shape = {X.shape}, Y shape = {Y.shape}")