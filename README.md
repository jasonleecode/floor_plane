# 户型图转机器人地图和路径规划系统

这是一个将户型图转换成机器人地图的程序，提供了多种路径规划算法，支持图形界面操作，可以在地图上设置起点和终点，然后生成规划路径。系统包含最短路径算法和适用于扫地机器人的清扫路径算法。

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

## 界面功能说明

### 主界面控件
- **Floor Plan**: 显示当前选择的户型图文件名
- **Select Image**: 打开文件选择对话框
- **Algorithm**: 选择路径规划算法（默认Dijkstra）
- **Load Map**: 加载和处理选中的户型图
- **Generate Path**: 生成路径
- **Clear Path**: 清除当前路径
- **Reset**: 重置所有设置

### 地图显示
- **灰色背景**: 原始户型图
- **红色圆圈**: 障碍物（带编号）
- **绿色圆点**: 起点
- **红色圆点**: 终点
- **蓝色线条**: 规划路径

### 操作说明
- **左键点击**: 设置起点
- **右键点击**: 设置终点
- **算法切换**: 使用下拉菜单选择算法
- **状态显示**: 底部状态栏显示当前操作状态

## 算法选择指南

### 最短路径规划
- **Dijkstra**: 默认算法，保证最短路径，适合大多数情况
- **A***: 启发式搜索，通常比Dijkstra更快
- **JPS**: 栅格地图的高效搜索

### 复杂环境规划
- **RRT**: 高维空间，复杂约束
- **RRT***: 需要渐近最优解

### 清扫路径规划
- **Zigzag**: 标准扫地机器人清扫模式
- **Spiral**: 从中心向外螺旋清扫

## 参数调整

### 图像处理参数
- `obstacle_scale`: 障碍物缩放因子，控制障碍物大小（推荐0.3）
- `grid_size`: 栅格大小，影响路径精度和计算时间（默认1.0）

### 路径规划参数
- `sweep_width`: Zigzag清扫宽度（默认3.0）
- `spiral_step`: Spiral清扫步长（默认2.0）
- `max_iterations`: RRT算法最大迭代次数（默认2000）

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

## 使用技巧

1. **图像质量**: 确保户型图对比度足够，墙壁清晰可见
2. **起点终点**: 选择在白色可通行区域的点作为起点和终点
3. **算法选择**: Dijkstra适合大多数情况，RRT适合复杂环境
4. **障碍物调整**: 如果障碍物太多或太少，可以调整obstacle_scale参数
5. **路径质量**: 如果路径不够平滑，可以减小grid_size提高精度

## 故障排除

### 常见问题
1. **无法加载图像**: 检查图像格式是否支持（PNG/JPG/JPEG/BMP/TIFF）
2. **路径规划失败**: 检查起点终点是否在可通行区域（白色区域）
3. **处理速度慢**: 图像太大时可以先用图像编辑软件缩小
4. **路径质量差**: 尝试不同的算法或调整参数

### 错误信息
- "Please select an image file first!": 需要先选择图像文件
- "Map not loaded!": 需要先点击"Load Map"加载地图
- "No path found": 起点和终点之间可能被障碍物阻挡，尝试调整位置

## 扩展功能

- 支持多种图像格式
- 障碍物自动编号
- 图形界面操作
- 实时路径规划
- 多算法比较
- 清扫路径规划
- 可导出路径数据

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

MIT License