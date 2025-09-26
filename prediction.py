import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import time
from IPython.display import HTML

cpu = torch.device('cpu')
gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CPU device: {cpu}, GPU device: {gpu}")

# CUSTOMIZE MATRIX SIZE HERE
sizes = list(range(200, 2200, 200))  # 200x200 => 2000x2000
cpu_times = []
gpu_times = []

def multiply_and_time(N, device):
    A = torch.randn(N, N)
    B = torch.randn(N, N)
    A_device = A.to(device)
    B_device = B.to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()         # start point
    _ = torch.matmul(A_device, B_device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.time()           # end point
    return end - start

# add times for all sizes
for N in sizes:
    t_cpu = multiply_and_time(N, cpu)
    t_gpu = multiply_and_time(N, gpu)
    cpu_times.append(t_cpu)
    gpu_times.append(t_gpu)

# matplot setup
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlim(0, max(sizes))
ax.set_ylim(0, max(max(cpu_times), max(gpu_times))*1.2)  # base y-axis scale on pre-computed times
line_cpu, = ax.plot([], [], lw=3, color='skyblue', label='CPU', marker='o')
line_gpu, = ax.plot([], [], lw=3, color='orange', label='GPU', marker='o')
ax.set_xlabel('Matrix Size (N x N)', fontsize=12)
ax.set_ylabel('Multiplication Time (s)', fontsize=12)
ax.set_title('Matrix Multiplication: CPU vs GPU', fontsize=14)
ax.legend()
ax.grid(True)

texts = []

def animate(i):
    N = sizes[i]
    line_cpu.set_data(sizes[:i+1], cpu_times[:i+1])
    line_gpu.set_data(sizes[:i+1], gpu_times[:i+1])
    ax.set_title(f'Matrix Multiplication: CPU vs GPU\nCurrent size: {N}x{N}', fontsize=14)

    # remove previous text objects once done with them
    for text in texts:
        text.remove()
    texts.clear()

    # add new text for current speed comparison
    speedup_text = ax.text(0.7*max(sizes), 0.9*ax.get_ylim()[1], f'Speedup: {cpu_times[i]/gpu_times[i]:.2f}×', fontsize=12, color='green')
    texts.append(speedup_text)

    return line_cpu, line_gpu, speedup_text # return new text object

anim = FuncAnimation(fig, animate, frames=len(sizes), interval=1000, blit=True)

# export gif
gif_filename = 'matrix_multiplication2.gif'
anim.save(gif_filename, writer=PillowWriter(fps=1))
print(f'GIF saved as {gif_filename}')