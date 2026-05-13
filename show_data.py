import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from dataclasses import dataclass
import datetime
import random
import string
import pickle
import os
import glob
import tkinter as tk
from tkinter import Listbox, END, filedialog
from GL import NetworkData

import sys, os
if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(__file__)

pkl_files = glob.glob(os.path.join(base_dir, 'data', '*.pkl'))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def generate_simulated_data(filename='network_data.pkl', N=12, months=60):
    start_date = datetime.date(2020, 1, 1)
    times = [start_date + pd.DateOffset(months=i) for i in range(months)]
    times = [t.date() for t in times]

    names = []
    for _ in range(N):
        length = random.randint(2, 4)
        name = ''.join(random.choices(string.ascii_uppercase, k=length))
        names.append(name)

    data_array = []
    current_jaccard = 0.5
    current_l1 = 0.5

    for t in times:
        step = np.random.normal(0, 0.05)
        current_jaccard = np.clip(current_jaccard + step, 0, 1)
        l1_step = np.random.normal(0, 0.02)
        current_l1 = np.clip(current_l1 + l1_step, 0.001, 2.0)

        A = np.random.choice([0, 1], size=(N, N), p=[0.7, 0.3])
        np.fill_diagonal(A, 0)
        X = np.random.randn(N, N)
        Theta = (X + X.T) / 2

        data_array.append(NetworkData(
            time=t,
            jaccard_index=current_jaccard,
            A=A,
            Theta=Theta,
            l1_penalty=current_l1
        ))

    with open(filename, 'wb') as f:
        pickle.dump({'name': names, 'data_array': data_array}, f)


def get_colored_adjacency(A, Theta):
    #返回形状 (N, N, 3) 的 RGB 数组，A==1 时根据Theta正负着色：负-红，正-蓝，其他为白色
    N = A.shape[0]
    rgb = np.ones((N, N, 3))  # 白色背景 [1,1,1]
    for i in range(N):
        for j in range(N):
            if A[i, j] == 1:
                if Theta[i, j] < 0:
                    rgb[i, j] = [1, 0, 0]   # 红色
                elif Theta[i, j] > 0:
                    rgb[i, j] = [0, 0, 1]   # 蓝色
                # Theta == 0 时保持白色
    return rgb

def run_ui():
    current_folder = os.path.join(base_dir, 'data')
    pkl_files = glob.glob(os.path.join(current_folder, '*.pkl'))
    cbar = None
    if not pkl_files:
        print("当前目录没有 .pkl 文件")
        default_file = os.path.join(current_folder, 'financial_network_data.pkl')
        generate_simulated_data(filename=default_file, N=15, months=72)
        pkl_files = [default_file]

    current_file = pkl_files[0]  

    # 加载初始文件
    def load_pkl(filename):
        with open(filename, 'rb') as f:
            raw = pickle.load(f)
        clean_array = []
        for item in raw['data_array']:
            if isinstance(item, dict):
                # 从字典重建 NetworkData
                clean_array.append(NetworkData(
                    time=item['time'],
                    jaccard_index=item['jaccard_index'],
                    A=item['A'],
                    Theta=item['Theta'],
                    l1_penalty=item['l1_penalty']
                ))
            else:
                clean_array.append(item)
        raw['data_array'] = clean_array
        return raw

    data = load_pkl(current_file)
    names = data['name']
    data_array = data['data_array']
    N = len(names)

    times = [d.time for d in data_array]
    time_strs = [t.strftime('%Y-%m') for t in times]
    jaccards = [d.jaccard_index for d in data_array]
    l1_penalties = [d.l1_penalty for d in data_array]
    global_max_abs = max(max(abs(d.Theta.min()), abs(d.Theta.max())) for d in data_array)

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(14, 9))
    fig.canvas.manager.set_window_title("GL演化可视化")

    gs = GridSpec(3, 2, height_ratios=[1.2, 3, 0.3], bottom=0.15, hspace=0.4, wspace=0.2)
    ax_line = fig.add_subplot(gs[0, :])
    ax_A = fig.add_subplot(gs[1, 0])
    ax_Theta = fig.add_subplot(gs[1, 1])
    ax_Theta_pos = ax_Theta.get_position()

    line_jaccard = ax_line.plot(times, jaccards, '-', color='#2b8cbe', linewidth=1.5, label='Jaccard Index')
    vline = ax_line.axvline(x=times[0], color='#de2d26', linestyle='--', linewidth=2)
    ax_line.set_ylabel("Jaccard Index", color='#2b8cbe')
    ax_line.set_ylim(-0.05, 1.05)
    ax_line.tick_params(axis='y', labelcolor='#2b8cbe')

    ax_line2 = ax_line.twinx()
    line_l1 = ax_line2.plot(times, l1_penalties, '-', color='#d95f02', linewidth=1.5, label='L1 Penalty')
    ax_line2.set_ylabel("L1 Penalty", color='#d95f02')
    ax_line2.tick_params(axis='y', labelcolor='#d95f02')

    lines = line_jaccard + line_l1
    labels = [l.get_label() for l in lines]
    ax_line.legend(lines, labels, loc='upper left')
    ax_line.set_title("Jaccard Index & L1 Penalty 演化序列", fontsize=12, fontweight='bold')

    curr_data = data_array[0]

    img_A = ax_A.imshow(get_colored_adjacency(curr_data.A, curr_data.Theta), interpolation='none', aspect='equal')
    ax_A.set_title(f"邻接矩阵 A ({time_strs[0]})", fontsize=11, y=-0.15)

    theta_raw = curr_data.Theta
    img_T = ax_Theta.imshow(theta_raw, cmap='coolwarm', vmin=-global_max_abs, vmax=global_max_abs, aspect='equal')
    ax_Theta.set_title(f"精度矩阵 Theta ({time_strs[0]})", fontsize=11, y=-0.15)
    cbar = fig.colorbar(img_T, ax=ax_Theta, fraction=0.046, pad=0.04)

    for ax in [ax_A, ax_Theta]:
        ax.set_box_aspect(1)
        ax.xaxis.set_ticks_position('top')
        ax.set_xticks(np.arange(N))
        ax.set_yticks(np.arange(N))
        ax.set_xticklabels(names, rotation=45, ha='left', fontsize=9)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xticks(np.arange(-.5, N, 1), minor=True)
        ax.set_yticks(np.arange(-.5, N, 1), minor=True)
        ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.grid(which='major', visible=False)

    ax_slider = plt.axes([0.15, 0.05, 0.5, 0.03])
    slider = Slider(
        ax=ax_slider, label='时间进度 ',
        valmin=0, valmax=len(times)-1,
        valinit=0, valstep=1, valfmt='%d', color='#2b8cbe'
    )

    ax_text = plt.axes([0.78, 0.045, 0.1, 0.04])
    textbox = TextBox(ax_text, '年月 (YYYY-MM): ', textalignment='center')

    ax_button = plt.axes([0.15, 0.11, 0.12, 0.04])
    btn_select = Button(ax_button, '选择文件', color='lightgoldenrodyellow', hovercolor='0.975')

    ax_folder_btn = plt.axes([0.28, 0.11, 0.12, 0.04])
    btn_folder = Button(ax_folder_btn, '选择文件夹', color='lightgoldenrodyellow', hovercolor='0.975')

    file_label = fig.text(0.42, 0.12, current_file, fontsize=9, color='#238b45')

    def update(val):
        idx = int(slider.val)
        d = data_array[idx]

        vline.set_xdata([times[idx], times[idx]])

        img_A.set_data(get_colored_adjacency(d.A, d.Theta))
        ax_A.set_title(f"邻接矩阵 A ({time_strs[idx]})", fontsize=11, y=-0.15)

        t_raw = d.Theta
        img_T.set_data(t_raw)
        img_T.set_clim(vmin=-global_max_abs, vmax=global_max_abs)
        ax_Theta.set_title(f"精度矩阵 Theta ({time_strs[idx]})", fontsize=11, y=-0.15)

        fig.canvas.draw_idle()

    def submit_text(text):
        if text in time_strs:
            idx = time_strs.index(text)
            slider.set_val(idx)
        else:
            print(f"输入的时间 '{text}' 不在数据范围内，已忽略。")
            textbox.set_val(time_strs[int(slider.val)])

    def load_new_file(new_file):
        nonlocal names, data_array, N, times, time_strs, jaccards, l1_penalties, current_file, vline
        nonlocal global_max_abs, img_A, img_T, cbar
        if new_file == current_file:
            return
        data = load_pkl(new_file)
        names = data['name']
        data_array = data['data_array']
        N = len(names)

        times = [d.time for d in data_array]
        time_strs = [t.strftime('%Y-%m') for t in times]
        jaccards = [d.jaccard_index for d in data_array]
        l1_penalties = [d.l1_penalty for d in data_array]
        current_file = new_file

        global_max_abs = max(max(abs(d.Theta.min()), abs(d.Theta.max())) for d in data_array)

        file_label.set_text(current_file)

        ax_line.clear()
        ax_line.plot(times, jaccards, '-', color='#2b8cbe', linewidth=1.5, label='Jaccard Index')
        vline = ax_line.axvline(x=times[0], color='#de2d26', linestyle='--', linewidth=2)
        ax_line.set_ylabel("Jaccard Index", color='#2b8cbe')
        ax_line.set_ylim(-0.05, 1.05)
        ax_line.tick_params(axis='y', labelcolor='#2b8cbe')

        ax_line2.clear()
        ax_line2.plot(times, l1_penalties, '-', color='#d95f02', linewidth=1.5, label='L1 Penalty')
        ax_line2.set_ylabel("L1 Penalty", color='#d95f02')
        ax_line2.tick_params(axis='y', labelcolor='#d95f02')

        ax_line.legend([ax_line.lines[0], ax_line2.lines[0]], ['Jaccard Index', 'L1 Penalty'], loc='upper left')
        ax_line.set_title("Jaccard Index & L1 Penalty 演化序列", fontsize=12, fontweight='bold')

        ax_A.clear()
        img_A = ax_A.imshow(get_colored_adjacency(data_array[0].A, data_array[0].Theta), interpolation='none', aspect='equal')
        ax_A.set_title(f"邻接矩阵 A ({time_strs[0]})", fontsize=11, y=-0.15)

        ax_Theta.clear()
        theta_raw = data_array[0].Theta
        img_T = ax_Theta.imshow(theta_raw, cmap='coolwarm', vmin=-global_max_abs, vmax=global_max_abs, aspect='equal')
        ax_Theta.set_title(f"精度矩阵 Theta ({time_strs[0]})", fontsize=11, y=-0.15)

        if cbar is not None:
            try:
                cbar.remove()
            except Exception:
                pass
            cbar = None

        ax_Theta.set_position(ax_Theta_pos)
        cbar = fig.colorbar(img_T, ax=ax_Theta, fraction=0.046, pad=0.04)

        # 重新设置两个矩阵图的坐标轴、网格和范围
        for ax in [ax_A, ax_Theta]:
            ax.set_box_aspect(1)
            ax.xaxis.set_ticks_position('top')
            ax.set_xticks(np.arange(N))
            ax.set_yticks(np.arange(N))
            ax.set_xticklabels(names, rotation=45, ha='left', fontsize=9)
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xticks(np.arange(-.5, N, 1), minor=True)
            ax.set_yticks(np.arange(-.5, N, 1), minor=True)
            ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)
            ax.tick_params(which='minor', bottom=False, left=False)
            ax.grid(which='major', visible=False)
            ax.set_xlim(-0.5, N - 0.5)
            ax.set_ylim(N - 0.5, -0.5)

        slider.valmin = 0
        slider.valmax = len(times) - 1
        slider.ax.set_xlim(0, len(times)-1)
        slider.set_val(0)

        fig.canvas.draw_idle()

    def select_folder(event):
        nonlocal current_folder, current_file
        folder = filedialog.askdirectory(initialdir=current_folder, title='选择数据文件夹')
        if not folder:
            return
        current_folder = folder
        new_files = glob.glob(os.path.join(current_folder, '*.pkl'))
        if not new_files:
            print(f"文件夹 '{current_folder}' 中没有 .pkl 文件")
            return
        chosen = new_files[0]
        load_new_file(chosen)

    def open_file_selector(event):
        nonlocal current_folder
        files = glob.glob(os.path.join(current_folder, '*.pkl'))
        if not files:
            print("当前目录没有 .pkl 文件！")
            return

        top = tk.Tk()
        top.title("选择数据文件")
        top.geometry("300x350")
        top.resizable(False, False)

        tk.Label(top, text="双击文件名加载，或选中后点‘加载’", pady=5).pack()

        listbox = Listbox(top, selectmode=tk.SINGLE)
        for f in files:
            listbox.insert(END, os.path.basename(f))
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        if current_file in files:
            idx = files.index(current_file)
            listbox.selection_set(idx)

        def on_load():
            selection = listbox.curselection()
            if selection:
                chosen = files[selection[0]]
                top.destroy()
                load_new_file(chosen)

        def on_double_click(event):
            on_load()

        btn = tk.Button(top, text="加载", command=on_load)
        btn.pack(pady=5)

        listbox.bind('<Double-Button-1>', on_double_click)

        top.mainloop()

    btn_select.on_clicked(open_file_selector)
    btn_folder.on_clicked(select_folder)

    slider.on_changed(update)
    textbox.on_submit(submit_text)
    textbox.set_val(time_strs[0])

    plt.show()


if __name__ == '__main__':
    run_ui()
