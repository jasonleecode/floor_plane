# 户型图转机器人地图和路径规划系统

这是一个将户型图转换成机器人地图的程序，提供了多种路径规划算法，支持图形界面操作，可以在地图上设置起点和终点，然后生成规划路径。系统包含最短路径算法和适用于扫地机器人的清扫路径算法。

<img width="1197" height="875" alt="1757319270390" src="https://github.com/user-attachments/assets/47e3c7fb-01d6-4695-9c48-7875142b2231" />


## 功能特性

### 🏠 户型图处理
- 自动加载PNG/JPG/JPEG/BMP/TIFF格式的户型图
- 智能提取墙壁和障碍物
- 支持图像缩放和预处理
- 自动生成机器人可用的栅格地图
- 障碍物自动编号显示

### 🛣️ 路径规划算法
- **A*** - 启发式搜索，最优路径
- **Dijkstra** - 保证最短路径（默认算法）
- **RRT** - 概率完备，适合复杂环境
- **RRT*** - 渐近最优版本
- **JPS** - A*的高效优化版本
- **Zigzag** - 扫地机器人专用清扫路径
- **Spiral** - 螺旋式清扫路径

### 🎯 应用场景
- 扫地机器人路径规划
- 室内机器人导航
- 户型图分析和处理
- 路径规划算法研究
- 教学演示和算法比较

### 🖱️ 交互式功能
- **图形界面操作** - 完全基于GUI，无需命令行参数
- **文件选择** - 支持文件对话框选择户型图
- **鼠标点击选择** - 左键选择起点，右键选择终点
- **实时路径规划** - 点击按钮生成路径
- **多算法支持** - 支持所有路径规划和清扫算法
- **可视化反馈** - 实时显示选择的点和规划结果
- **障碍物编号** - 每个障碍物都有清晰编号

## 快速开始

### 1. 图形界面程序（推荐）

```bash
# 直接运行图形界面程序
python interactive_path_planner.py

# 或使用启动脚本
./run_planner.sh
```

**操作流程：**
1. 启动程序后，点击"Select Image"选择户型图
2. 点击"Load Map"加载和处理地图
3. 左键点击地图设置起点
4. 右键点击地图设置终点
5. 点击"Generate Path"生成路径
6. 使用"Clear Path"清除路径或"Reset"重新开始

### 2. 程序化使用

```python
from script.house_plan_to_robot_map import HousePlanToRobotMap
from script.floor_plan.path_planning import Point

# 加载户型图
system = HousePlanToRobotMap("your_floor_plan.png", obstacle_scale=0.3)

# 设置起点和终点
start = Point(50, 50)
goal = Point(200, 200)

# 规划路径
path = system.plan_path(start, goal, 'Dijkstra')

# 可视化结果
system.visualize_path(start, goal, 'Dijkstra')
```

### 3. 清扫路径规划

```python
# Zigzag清扫路径
cleaning_path = system.plan_cleaning_path(start, 'Zigzag')
system.visualize_cleaning_path(start, 'Zigzag')

# Spiral清扫路径
spiral_path = system.plan_cleaning_path(start, 'Spiral')
system.visualize_cleaning_path(start, 'Spiral')
```

### 4. 综合测试

```bash
# 运行所有测试
python test/comprehensive_test.py

# 只运行交互式演示
python test/comprehensive_test.py --interactive
```

## 运行示例

### 图形界面程序
```bash
# 启动图形界面
python interactive_path_planner.py
```

### 综合测试
```bash
# 运行完整测试套件
python test/comprehensive_test.py
```

### 演示程序
```bash
# 交互式演示
python demo_interactive_planner.py
```

## 文件结构

```
floor_plane/
├── interactive_path_planner.py        # 主图形界面程序
├── demo_interactive_planner.py        # 演示程序
├── run_planner.sh                     # 启动脚本
├── script/
│   ├── floor_plan/
│   │   ├── path_planning.py          # 路径规划算法库
│   │   ├── floor_plan.py             # 楼层平面图处理
│   │   ├── floorplan_extraction.py   # 图像提取算法
│   │   └── utils.py                  # 工具函数
│   ├── house_plan_to_robot_map.py    # 主系统
│   └── floor_plan_runtime.py         # 运行时系统
├── test/
│   ├── comprehensive_test.py         # 综合测试文件
│   └── README.md                     # 测试说明
├── topdown_floors.png                # 示例户型图
├── topdown_floors2.png
├── topdown_floors3.png
└── README.md
```


## 依赖项

- Python 3.6+
- OpenCV (cv2)
- NumPy
- Matplotlib
- SciPy
- scikit-image
- tkinter (通常随Python安装)

## 安装依赖

```bash
pip install opencv-python numpy matplotlib scipy scikit-image
```


## 扩展功能

- 支持多种图像格式
- 障碍物自动编号
- 图形界面操作
- 实时路径规划
- 多算法比较
- 清扫路径规划
- 可导出路径数据

