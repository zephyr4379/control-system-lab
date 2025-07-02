import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, step
import pyswarms as ps
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 设置 matplotlib 后端（非交互）
import matplotlib
matplotlib.use('Agg')

# ---------------------------
# 1. 加载数据
# ---------------------------
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='utf-8')
    print("\n数据基本信息:")
    data.info()
    print("\n数据预览:", data.head())

    time = data['time'].values
    temperature = data['temperature'].values
    voltage = data['volte'].values

    return time, temperature, voltage

# ---------------------------
# 2. 模型验证
# ---------------------------
def simulate_transfer_function(K, T, L, time, t0):
    num = [K]
    den = [T, 1]
    sys = TransferFunction(num, den)
    t_step, y_step = step(sys, T=time)

    delay_idx = int(L // (time[1] - time[0])) if len(time) > 1 else 0
    y_delayed = np.roll(y_step, delay_idx)
    y_delayed[:delay_idx] = t0
    return y_delayed

# ---------------------------
# 3. PID 控制仿真
# ---------------------------
def pid_control(y_set, y_meas, Kp, Ki, Kd, dt, integral, prev_error):
    error = y_set - y_meas
    integral += error * dt
    derivative = (error - prev_error) / dt if dt != 0 else 0
    output = Kp * error + Ki * integral + Kd * derivative
    return output, integral, error

def simulate_pid(Kp, Ki, Kd, K, T, L, y_set=35, dt=1, sim_time=15000):
    time_sim = np.arange(0, sim_time, dt)
    y_sim = np.zeros_like(time_sim)
    u = np.zeros_like(time_sim)
    integral = 0
    prev_error = 0
    delay_buffer = np.zeros(int(L // dt)) if dt != 0 else np.zeros(1)

    for i, t in enumerate(time_sim):
        if i == 0:
            pid_out, integral, prev_error = pid_control(y_set, y_sim[i], Kp, Ki, Kd, dt, 0, 0)
        else:
            pid_out, integral, prev_error = pid_control(y_set, y_sim[i - 1], Kp, Ki, Kd, dt, integral, prev_error)

        pid_out = np.clip(pid_out, 0, 10)
        u[i] = pid_out

        if len(delay_buffer) > 0:
            delay_buffer = np.roll(delay_buffer, 1)
            delay_buffer[0] = pid_out
            u_delayed = delay_buffer[-1]
        else:
            u_delayed = pid_out

        if i == 0:
            y_sim[i] = 16.85  # 初始温度
        else:
            tau = T / dt
            y_sim[i] = y_sim[i - 1] + (u_delayed * K - y_sim[i - 1]) / tau

    error = np.abs(y_sim - y_set)
    itae = np.sum(time_sim * error)
    overshoot = np.max(y_sim) - y_set if np.max(y_sim) > y_set else 0

    tol = 0.02 * (y_set - 16.85)
    settling_time = sim_time
    for i in range(len(y_sim) - 200):
        window = y_sim[i:i + 200]
        if np.all(np.abs(window - y_set) < tol):
            settling_time = time_sim[i]
            break

    return time_sim, y_sim, u, itae, overshoot, settling_time

# ---------------------------
# 4. 优化目标函数
# ---------------------------
def objective_function(pid_params, K, T, L, y_set, dt, sim_time):
    n_particles = pid_params.shape[0]
    j = np.zeros(n_particles)
    for i in range(n_particles):
        Kp, Ki, Kd = pid_params[i]
        _, y_sim, _, itae, overshoot, _ = simulate_pid(Kp, Ki, Kd, K, T, L, y_set, dt, sim_time)
        penalty = overshoot * 100
        j[i] = itae + penalty
    return j

# ---------------------------
# 5. 主程序
# ---------------------------
if __name__ == "__main__":
    file_path = r"C:\\Users\\zephyr\\Documents\\WeChat Files\\wxid_85ieax35pwyg22\\FileStorage\\File\\2025-06\\ai control system\\ai control system\\B 任务数据集.csv"
    time, temperature, voltage = load_data(file_path)

    print("\n使用拟合模型参数: K=3.45, T=2989s, L=53s")
    K = 3.45
    T = 2989.0
    L = 53.0

    y_identified = simulate_transfer_function(K, T, L, time, temperature[0])

    plt.figure(figsize=(10, 5))
    plt.plot(time, temperature, label='Original Temperature')
    plt.plot(time, y_identified, 'r--', label='Identified Model')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.title("模型验证图")
    plt.savefig("model_validation.png")

    print("\n开始PID优化...")
    bounds = ([0.1, 0.001, 0.01], [5.0, 0.1, 1.0])
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=3, options=options, bounds=bounds)

    cost, optimal_params = optimizer.optimize(
        objective_function,
        iters=30,
        K=K, T=T, L=L, y_set=35, dt=1, sim_time=15000
    )

    Kp_opt, Ki_opt, Kd_opt = optimal_params
    print(f"优化PID参数：Kp={Kp_opt:.3f}, Ki={Ki_opt:.3f}, Kd={Kd_opt:.3f}")

    time_sim, y_sim, u_sim, itae, overshoot, settling_time = simulate_pid(Kp_opt, Ki_opt, Kd_opt, K, T, L)
    steady_state_error = np.abs(y_sim[-1] - 35)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_sim, y_sim, label='Temperature')
    plt.axhline(35, color='r', linestyle='--', label='Setpoint')
    plt.ylabel('Temperature (°C)')
    plt.title('PID 控制结果')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time_sim, u_sim, label='Voltage')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('pid_control_result.png')

    print("\n控制性能指标：")
    print(f"ITAE: {itae:.2f}")
    print(f"超调量: {overshoot:.2f}°C")
    print(f"调节时间: {settling_time:.2f}s")
    print(f"稳态误差: {steady_state_error:.2f}°C")
