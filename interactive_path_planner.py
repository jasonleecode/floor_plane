#!/usr/bin/env python3

"""
交互式路径规划程序
支持命令行参数指定户型图和算法，在界面上设置起点和终点，按P键生成路径
"""

import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import glob

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 禁用字体警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 添加script目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'script'))

from house_plan_to_robot_map import HousePlanToRobotMap
from floor_plan.path_planning import Point

class InteractivePathPlanner:
    """交互式路径规划器"""
    
    def __init__(self):
        self.image_path = None
        self.algorithm = 'Dijkstra'
        self.converter = None
        self.start_point = None
        self.goal_point = None
        self.current_path = None
        
        # 强制设置中文字体
        import matplotlib
        import matplotlib.font_manager as fm
        
        # 清除字体缓存
        try:
            fm._rebuild()
        except AttributeError:
            # 如果_rebuild不存在，尝试其他方法
            pass
        
        # 设置字体
        matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK', 'SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("Interactive Path Planner")
        self.root.geometry("1200x800")
        
        # 创建matplotlib图形
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.canvas = None
        
        # 初始化界面
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 第一行：文件选择和算法选择
        top_frame = ttk.Frame(control_frame)
        top_frame.pack(fill=tk.X, pady=(0, 5))
        
        # 文件选择
        ttk.Label(top_frame, text="Floor Plan:").pack(side=tk.LEFT, padx=(0, 5))
        self.image_var = tk.StringVar(value="No file selected")
        self.image_label = ttk.Label(top_frame, textvariable=self.image_var, foreground="blue")
        self.image_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(top_frame, text="Select Image", command=self.select_image).pack(side=tk.LEFT, padx=(0, 20))
        
        # 算法选择
        ttk.Label(top_frame, text="Algorithm:").pack(side=tk.LEFT, padx=(0, 5))
        self.algorithm_var = tk.StringVar(value=self.algorithm)
        algorithm_combo = ttk.Combobox(top_frame, textvariable=self.algorithm_var, 
                                     values=['A*', 'Dijkstra', 'RRT', 'RRT*', 'JPS'], 
                                     state='readonly', width=10)
        algorithm_combo.pack(side=tk.LEFT, padx=(0, 20))
        algorithm_combo.bind('<<ComboboxSelected>>', self.on_algorithm_change)
        
        # 第二行：操作按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(button_frame, text="Load Map", command=self.load_map).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Generate Path", command=self.plan_path).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Clear Path", command=self.clear_path).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Reset", command=self.reset_all).pack(side=tk.LEFT, padx=(0, 5))
        
        # 状态标签
        self.status_var = tk.StringVar(value="Please select a floor plan image and load the map")
        status_label = ttk.Label(control_frame, textvariable=self.status_var, foreground="blue")
        status_label.pack(fill=tk.X, pady=(5, 0))
        
        # 创建matplotlib画布
        self.canvas = FigureCanvasTkAgg(self.fig, main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 绑定事件
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        # 显示初始提示
        self.display_initial_message()
    
    def select_image(self):
        """选择图像文件"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Floor Plan Image",
            filetypes=filetypes,
            initialdir=os.getcwd()
        )
        
        if filename:
            self.image_path = filename
            self.image_var.set(os.path.basename(filename))
            self.status_var.set(f"Image selected: {os.path.basename(filename)}. Click 'Load Map' to load it.")
            # 清除之前的地图和路径
            self.converter = None
            self.start_point = None
            self.goal_point = None
            self.current_path = None
            self.display_initial_message()
    
    def display_initial_message(self):
        """显示初始提示信息"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Please select a floor plan image\nand click "Load Map" to begin', 
                    ha='center', va='center', fontsize=16, 
                    transform=self.ax.transAxes)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("Interactive Path Planner", fontsize=14, weight='bold')
        self.canvas.draw()
        
    def load_map(self):
        """加载地图"""
        if not self.image_path:
            messagebox.showwarning("Warning", "Please select an image file first!")
            return
            
        try:
            self.status_var.set("Loading map...")
            self.root.update()
            
            self.converter = HousePlanToRobotMap(self.image_path, obstacle_scale=0.3)
            self.display_map()
            
            self.status_var.set("Map loaded! Click to set start and goal points, then click 'Generate Path'")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load map: {str(e)}")
            self.status_var.set("Map loading failed")
    
    def display_map(self):
        """显示地图"""
        if not self.converter:
            return
            
        self.ax.clear()
        
        # 显示原始图像
        if self.converter.original_image is not None:
            self.ax.imshow(self.converter.original_image, cmap='gray')
        
        # 显示障碍物
        for i, (center, radius) in enumerate(self.converter.planners[self.algorithm].obstacles):
            circle = patches.Circle((center.x, center.y), radius, 
                                  color='red', alpha=0.6, label='Obstacles' if i == 0 else "")
            self.ax.add_patch(circle)
            
            # 添加障碍物编号
            self.ax.text(center.x, center.y, str(i+1), 
                        ha='center', va='center', fontsize=10, 
                        color='white', weight='bold',
                        bbox=dict(boxstyle='circle', facecolor='red', alpha=0.8, pad=0.1))
        
        # 显示起点
        if self.start_point:
            self.ax.plot(self.start_point.x, self.start_point.y, 'go', 
                        markersize=12, label='Start', markeredgecolor='darkgreen', 
                        markeredgewidth=2)
            self.ax.text(self.start_point.x + 5, self.start_point.y + 5, 
                        f'Start\n({self.start_point.x:.1f}, {self.start_point.y:.1f})',
                        fontsize=10, color='green', weight='bold')
        
        # 显示终点
        if self.goal_point:
            self.ax.plot(self.goal_point.x, self.goal_point.y, 'ro', 
                        markersize=12, label='Goal', markeredgecolor='darkred', 
                        markeredgewidth=2)
            self.ax.text(self.goal_point.x + 5, self.goal_point.y + 5, 
                        f'Goal\n({self.goal_point.x:.1f}, {self.goal_point.y:.1f})',
                        fontsize=10, color='red', weight='bold')
        
        # 显示路径
        if self.current_path and len(self.current_path) > 1:
            path_x = [p.x for p in self.current_path]
            path_y = [p.y for p in self.current_path]
            self.ax.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.8, label='Planned Path')
            self.ax.plot(path_x, path_y, 'bo', markersize=4, alpha=0.6)
        
        # 设置标题和标签
        self.ax.set_title(f'Interactive Path Planning - {self.algorithm} Algorithm', fontsize=14, weight='bold')
        self.ax.set_xlabel('X Coordinate', fontsize=12)
        self.ax.set_ylabel('Y Coordinate', fontsize=12)
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        
        # 设置坐标轴范围
        if self.converter.original_image is not None:
            h, w = self.converter.original_image.shape[:2]
            self.ax.set_xlim(0, w)
            self.ax.set_ylim(h, 0)  # 图像坐标系Y轴向下
        
        self.canvas.draw()
    
    def on_click(self, event):
        """处理鼠标点击事件"""
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:  # 左键 - 设置起点
            self.start_point = Point(event.xdata, event.ydata)
            self.status_var.set(f"Start point set: ({self.start_point.x:.1f}, {self.start_point.y:.1f})")
            self.display_map()
            
        elif event.button == 3:  # 右键 - 设置终点
            self.goal_point = Point(event.xdata, event.ydata)
            self.status_var.set(f"Goal point set: ({self.goal_point.x:.1f}, {self.goal_point.y:.1f})")
            self.display_map()
    
    def reset_all(self):
        """重置所有"""
        self.image_path = None
        self.converter = None
        self.start_point = None
        self.goal_point = None
        self.current_path = None
        self.image_var.set("No file selected")
        self.status_var.set("Please select a floor plan image and load the map")
        self.display_initial_message()
    
    def on_algorithm_change(self, event):
        """处理算法改变事件"""
        self.algorithm = self.algorithm_var.get()
        self.status_var.set(f"Algorithm switched to: {self.algorithm}")
        # 清除当前路径
        self.current_path = None
        self.display_map()
    
    def plan_path(self):
        """规划路径"""
        if not self.start_point or not self.goal_point:
            messagebox.showwarning("Warning", "Please set start and goal points first!")
            return
        
        if not self.converter:
            messagebox.showerror("Error", "Map not loaded!")
            return
        
        try:
            self.status_var.set("Planning path...")
            self.root.update()
            
            # 检查起点和终点是否在可通行区域
            if not self.converter.is_point_in_navigable_area(self.start_point):
                messagebox.showwarning("Warning", f"Start point ({self.start_point.x:.1f}, {self.start_point.y:.1f}) is not in navigable area!")
                return
            
            if not self.converter.is_point_in_navigable_area(self.goal_point):
                messagebox.showwarning("Warning", f"Goal point ({self.goal_point.x:.1f}, {self.goal_point.y:.1f}) is not in navigable area!")
                return
            
            # 规划路径
            path = self.converter.plan_path(self.start_point, self.goal_point, self.algorithm)
            
            if path and len(path) > 1:
                self.current_path = path
                length = sum(path[i].distance_to(path[i+1]) for i in range(len(path)-1))
                self.status_var.set(f"Path planning successful! Length: {length:.2f}, Points: {len(path)}")
                self.display_map()
            else:
                self.current_path = None
                messagebox.showwarning("Warning", f"No path found using {self.algorithm} algorithm!")
                self.status_var.set("No path found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Path planning failed: {str(e)}")
            self.status_var.set("Path planning failed")
    
    def clear_path(self):
        """清除路径"""
        self.current_path = None
        self.display_map()
        self.status_var.set("Path cleared")
    
    
    def run(self):
        """运行程序"""
        # 显示使用说明
        instructions = """
Instructions:
• Select Image: Choose a floor plan image file
• Load Map: Process the selected image and load the map
• Left click: Set start point
• Right click: Set goal point
• Generate Path: Plan path using selected algorithm
• Clear Path: Remove current path
• Reset: Start over with new image
• Algorithm: Choose path planning algorithm
        """
        messagebox.showinfo("Instructions", instructions)
        
        self.root.mainloop()

def main():
    """主函数"""
    try:
        # 创建并运行交互式路径规划器
        planner = InteractivePathPlanner()
        planner.run()
    except Exception as e:
        print(f"Program error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
