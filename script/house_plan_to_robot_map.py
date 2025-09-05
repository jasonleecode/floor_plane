#! /usr/bin/env python3

"""
户型图转机器人地图和路径规划系统
支持将户型图转换为机器人可用的地图，并提供多种路径规划算法
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional
import os

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'script'))

from floor_plan.path_planning import *
from floor_plan.floorplan_extraction import load_floor_plan

class HousePlanToRobotMap:
    """户型图转机器人地图系统"""
    
    def __init__(self, image_path: str, scale_factor: float = 1.0, obstacle_scale: float = 1.0):
        """
        初始化系统
        
        Args:
            image_path: 户型图路径
            scale_factor: 缩放因子，用于调整地图大小
            obstacle_scale: 障碍物缩放因子，用于调整障碍物大小
        """
        self.image_path = image_path
        self.scale_factor = scale_factor
        self.obstacle_scale = obstacle_scale
        self.original_image = None
        self.processed_map = None
        self.bounds = None
        self.obstacles = []
        self.rooms = []
        
        # 加载和处理户型图
        self._load_and_process_image()
        
        # 初始化路径规划器
        self._init_path_planners()
    
    def _load_and_process_image(self):
        """加载和处理户型图"""
        print(f"加载户型图: {self.image_path}")
        
        # 加载图像
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"无法加载图像: {self.image_path}")
        
        # 转换为灰度图
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # 二值化处理
        _, binary = cv2.threshold(gray, 250, 1, cv2.THRESH_BINARY)
        
        # 缩放图像
        if self.scale_factor != 1.0:
            new_width = int(binary.shape[1] * self.scale_factor)
            new_height = int(binary.shape[0] * self.scale_factor)
            binary = cv2.resize(binary, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            _, binary = cv2.threshold(binary, 0.5, 1, cv2.THRESH_BINARY)
        
        self.processed_map = binary
        
        # 设置边界
        height, width = binary.shape
        self.bounds = (Point(0, 0), Point(width, height))
        
        print(f"图像尺寸: {width} x {height}")
        print(f"边界: {self.bounds[0]} 到 {self.bounds[1]}")
    
    def _init_path_planners(self):
        """初始化路径规划器"""
        self.planners = {
            'A*': AStar(grid_size=1.0),
            'Dijkstra': Dijkstra(grid_size=1.0),
            'RRT': RRT(grid_size=1.0, step_size=2.0, max_iterations=2000),
            'RRT*': RRTStar(grid_size=1.0, step_size=2.0, max_iterations=2000),
            'JPS': JPS(grid_size=1.0),
            'Zigzag': ZigzagPlanner(grid_size=1.0, sweep_width=3.0),
            'Spiral': SpiralPlanner(grid_size=1.0, spiral_step=2.0)
        }
        
        # 为每个路径规划器设置墙壁检查函数
        for planner in self.planners.values():
            planner.wall_checker = self.is_point_in_navigable_area
        
        # 为所有规划器添加障碍物
        self._extract_obstacles_from_map()
    
    def _extract_obstacles_from_map(self):
        """从地图中提取障碍物 - 在可通行区域内放置障碍物"""
        print("提取障碍物（在可通行区域内）...")
        
        # 在processed_map中，白色区域（值为1）是可通行区域，黑色区域（值为0）是墙壁
        # 障碍物应该放在可通行区域内，模拟房间内的家具等障碍物
        
        # 找到可通行区域（白色区域）
        navigable_area = (self.processed_map == 1).astype(np.uint8)
        
        # 使用形态学操作清理可通行区域
        kernel = np.ones((3, 3), np.uint8)
        navigable_area = cv2.morphologyEx(navigable_area, cv2.MORPH_CLOSE, kernel)
        
        # 找到可通行区域的轮廓
        contours, _ = cv2.findContours(navigable_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacle_count = 0
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 只处理足够大的可通行区域
            min_area = 100  # 最小面积阈值
            
            if area > min_area:
                # 计算轮廓的边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                
                # 在可通行区域内放置障碍物
                step_size = max(50, int(min(w, h) / 4))  # 障碍物间距，增加间距
                
                for i in range(step_size, w - step_size, step_size):
                    for j in range(step_size, h - step_size, step_size):
                        # 检查这个位置是否在可通行区域内
                        if (x + i < navigable_area.shape[1] and y + j < navigable_area.shape[0] and 
                            navigable_area[y + j, x + i] == 1):
                            
                            center = Point(x + i, y + j)
                            radius = step_size / 4 * self.obstacle_scale
                            
                            # 确保半径在合理范围内
                            radius = max(5, min(radius, 25))
                            
                            # 检查障碍物是否与墙壁重叠
                            if self._is_obstacle_valid(center, radius):
                                for planner in self.planners.values():
                                    planner.add_obstacle(center, radius)
                                
                                self.obstacles.append((center, radius))
                                obstacle_count += 1
        
        print(f"生成了 {obstacle_count} 个障碍物")
        
        # 如果没有生成障碍物，在可通行区域中心添加一些测试障碍物
        if obstacle_count == 0:
            print("未找到合适的可通行区域，添加测试障碍物")
            # 尝试在地图中心区域添加障碍物
            center_x, center_y = self.bounds[1].x * 0.5, self.bounds[1].y * 0.5
            test_centers = [
                Point(center_x * 0.3, center_y * 0.3),
                Point(center_x * 0.7, center_y * 0.3),
                Point(center_x * 0.3, center_y * 0.7),
                Point(center_x * 0.7, center_y * 0.7),
            ]
            
            for center in test_centers:
                if self.is_point_in_navigable_area(center):
                    radius = 15
                    for planner in self.planners.values():
                        planner.add_obstacle(center, radius)
                    self.obstacles.append((center, radius))
                    obstacle_count += 1
            print(f"添加了 {obstacle_count} 个测试障碍物")
    
    def _is_obstacle_valid(self, center: Point, radius: float) -> bool:
        """检查障碍物是否有效（不与墙壁重叠）"""
        # 检查障碍物中心是否在可通行区域
        if not self.is_point_in_navigable_area(center):
            return False
        
        # 检查障碍物边界是否与墙壁重叠
        # 简单检查：在障碍物周围采样几个点
        sample_points = [
            Point(center.x - radius, center.y),
            Point(center.x + radius, center.y),
            Point(center.x, center.y - radius),
            Point(center.x, center.y + radius),
        ]
        
        for point in sample_points:
            if not self.is_point_in_navigable_area(point):
                return False
        
        return True
    
    def is_point_in_navigable_area(self, point: Point) -> bool:
        """检查点是否在可通行区域内"""
        x, y = int(point.x), int(point.y)
        
        # 检查边界
        if (x < 0 or x >= self.processed_map.shape[1] or 
            y < 0 or y >= self.processed_map.shape[0]):
            return False
        
        # 检查是否在白色区域（可通行）
        return self.processed_map[y, x] == 1
    
    def find_valid_start_goal(self, start: Point, goal: Point) -> tuple:
        """找到有效的起点和终点"""
        # 如果起点不在可通行区域，寻找最近的可通行点
        if not self.is_point_in_navigable_area(start):
            start = self._find_nearest_navigable_point(start)
            print(f"调整起点到: ({start.x:.1f}, {start.y:.1f})")
        
        # 如果终点不在可通行区域，寻找最近的可通行点
        if not self.is_point_in_navigable_area(goal):
            goal = self._find_nearest_navigable_point(goal)
            print(f"调整终点到: ({goal.x:.1f}, {goal.y:.1f})")
        
        return start, goal
    
    def _find_nearest_navigable_point(self, point: Point) -> Point:
        """找到最近的可通行点"""
        # 在周围搜索可通行点
        search_radius = 50
        step = 5
        
        for radius in range(step, search_radius, step):
            for dx in range(-radius, radius + 1, step):
                for dy in range(-radius, radius + 1, step):
                    if dx*dx + dy*dy <= radius*radius:
                        test_point = Point(point.x + dx, point.y + dy)
                        if self.is_point_in_navigable_area(test_point):
                            return test_point
        
        # 如果找不到，返回地图中心
        center = Point(self.bounds[1].x / 2, self.bounds[1].y / 2)
        print(f"未找到可通行点，使用地图中心: ({center.x:.1f}, {center.y:.1f})")
        return center
    
    def plan_path(self, start: Point, goal: Point, algorithm: str = 'A*') -> List[Point]:
        """
        规划路径
        
        Args:
            start: 起点
            goal: 终点
            algorithm: 算法名称
            
        Returns:
            路径点列表
        """
        if algorithm not in self.planners:
            raise ValueError(f"未知算法: {algorithm}")
        
        print(f"使用 {algorithm} 算法规划路径...")
        print(f"原始起点: {start}")
        print(f"原始终点: {goal}")
        
        # 调整起点和终点到可通行区域
        start, goal = self.find_valid_start_goal(start, goal)
        
        print(f"调整后起点: {start}")
        print(f"调整后终点: {goal}")
        
        path = self.planners[algorithm].plan(start, goal, self.bounds)
        
        if path:
            # 检查路径是否在边界内
            valid_path = []
            for point in path:
                if (0 <= point.x <= self.bounds[1].x and 
                    0 <= point.y <= self.bounds[1].y):
                    valid_path.append(point)
                else:
                    print(f"警告: 路径点 ({point.x:.1f}, {point.y:.1f}) 超出边界!")
            
            if len(valid_path) != len(path):
                print(f"过滤了 {len(path) - len(valid_path)} 个超出边界的路径点")
                path = valid_path
            
            if path:
                length = sum(path[i].distance_to(path[i+1]) for i in range(len(path)-1))
                print(f"找到路径，长度: {length:.2f}, 点数: {len(path)}")
                print(f"路径起点: ({path[0].x:.1f}, {path[0].y:.1f})")
                print(f"路径终点: ({path[-1].x:.1f}, {path[-1].y:.1f})")
            else:
                print("过滤后路径为空")
        else:
            print("未找到路径")
        
        return path
    
    def plan_cleaning_path(self, start: Point, algorithm: str = 'Zigzag') -> List[Point]:
        """
        规划清扫路径
        
        Args:
            start: 起点
            algorithm: 清扫算法 ('Zigzag' 或 'Spiral')
            
        Returns:
            清扫路径点列表
        """
        if algorithm not in ['Zigzag', 'Spiral']:
            raise ValueError("清扫算法必须是 'Zigzag' 或 'Spiral'")
        
        print(f"使用 {algorithm} 算法规划清扫路径...")
        print(f"原始起点: {start}")
        
        # 调整起点到可通行区域
        if not self.is_point_in_navigable_area(start):
            start = self._find_nearest_navigable_point(start)
            print(f"调整起点到: ({start.x:.1f}, {start.y:.1f})")
        
        # 选择一个合适的终点（比如起点附近）
        goal = Point(start.x + 5, start.y + 5)
        
        path = self.planners[algorithm].plan(start, goal, self.bounds)
        
        if path:
            length = sum(path[i].distance_to(path[i+1]) for i in range(len(path)-1))
            print(f"生成清扫路径，长度: {length:.2f}, 点数: {len(path)}")
        else:
            print("未生成清扫路径")
        
        return path
    
    def visualize_map_conversion(self):
        """可视化地图转换过程"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 原始图像
        ax1 = axes[0, 0]
        if self.original_image is not None:
            ax1.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
            ax1.set_title('1. Original Floor Plan')
        else:
            ax1.text(0.5, 0.5, 'No Original Image', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('1. Original Floor Plan')
        ax1.axis('off')
        
        # 2. 灰度图
        ax2 = axes[0, 1]
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            ax2.imshow(gray, cmap='gray')
            ax2.set_title('2. Grayscale Image')
        else:
            ax2.text(0.5, 0.5, 'No Grayscale Image', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('2. Grayscale Image')
        ax2.axis('off')
        
        # 3. 二值化结果
        ax3 = axes[0, 2]
        ax3.imshow(self.processed_map, cmap='gray')
        ax3.set_title('3. Binary Map\n(White=Navigable, Black=Walls)')
        ax3.axis('off')
        
        # 4. 墙壁检测
        ax4 = axes[1, 0]
        walls = (self.processed_map == 0).astype(np.uint8)
        ax4.imshow(walls, cmap='gray')
        ax4.set_title('4. Wall Detection\n(Black=Walls)')
        ax4.axis('off')
        
        # 5. 障碍物分布 - 只显示在可通行区域的障碍物
        ax5 = axes[1, 1]
        ax5.imshow(self.processed_map, cmap='gray', extent=[0, self.bounds[1].x, self.bounds[1].y, 0])
        
        # 显示障碍物 - 只显示在可通行区域的
        valid_obstacles = []
        for i, (center, radius) in enumerate(self.obstacles):
            # 检查障碍物中心是否在可通行区域
            if self.is_point_in_navigable_area(center):
                circle = patches.Circle(center.to_tuple(), radius, 
                                      color='red', alpha=0.6)
                ax5.add_patch(circle)
                # 添加障碍物编号
                ax5.text(center.x, center.y, str(len(valid_obstacles)+1), ha='center', va='center', 
                        color='white', fontsize=8, weight='bold')
                valid_obstacles.append((center, radius))
        
        ax5.set_xlim(0, self.bounds[1].x)
        ax5.set_ylim(0, self.bounds[1].y)
        ax5.set_title(f'5. Obstacle Distribution\n({len(valid_obstacles)} valid obstacles)')
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        ax5.grid(True, alpha=0.3)
        
        # 6. 最终地图
        ax6 = axes[1, 2]
        ax6.imshow(self.processed_map, cmap='gray', extent=[0, self.bounds[1].x, self.bounds[1].y, 0])
        
        # 显示有效障碍物
        for center, radius in valid_obstacles:
            circle = patches.Circle(center.to_tuple(), radius, 
                                  color='red', alpha=0.4)
            ax6.add_patch(circle)
        
        # 添加一些测试点
        test_points = [
            Point(self.bounds[1].x * 0.2, self.bounds[1].y * 0.2),
            Point(self.bounds[1].x * 0.5, self.bounds[1].y * 0.5),
            Point(self.bounds[1].x * 0.8, self.bounds[1].y * 0.8)
        ]
        
        for i, point in enumerate(test_points):
            if self.is_point_in_navigable_area(point):
                ax6.plot(point.x, point.y, 'go', markersize=8)
                ax6.text(point.x, point.y+10, f'Valid{i+1}', ha='center', fontsize=8)
            else:
                ax6.plot(point.x, point.y, 'ro', markersize=8)
                ax6.text(point.x, point.y+10, f'Invalid{i+1}', ha='center', fontsize=8)
        
        ax6.set_xlim(0, self.bounds[1].x)
        ax6.set_ylim(0, self.bounds[1].y)
        ax6.set_title('6. Final Map\n(Green=Valid Points, Red=Invalid Points)')
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 打印障碍物统计信息
        print(f"总障碍物数量: {len(self.obstacles)}")
        print(f"有效障碍物数量: {len(valid_obstacles)}")
        print(f"无效障碍物数量: {len(self.obstacles) - len(valid_obstacles)}")
    
    def visualize_coordinate_mapping(self):
        """可视化坐标映射关系"""
        print("\n=== 坐标映射可视化 ===")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 原始图像
        ax1 = axes[0]
        if self.original_image is not None:
            ax1.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
            ax1.set_title('1. Original Image\n(Image Coordinates)')
        else:
            ax1.text(0.5, 0.5, 'No Original Image', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('1. Original Image')
        ax1.axis('off')
        
        # 2. 处理后的地图
        ax2 = axes[1]
        ax2.imshow(self.processed_map, cmap='gray', extent=[0, self.bounds[1].x, 0, self.bounds[1].y])
        ax2.set_title('2. Processed Map\n(World Coordinates)')
        ax2.set_xlabel('X (World)')
        ax2.set_ylabel('Y (World)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 叠加显示
        ax3 = axes[2]
        if self.original_image is not None:
            # 显示原始图像，使用正确的坐标范围
            orig_height, orig_width = self.original_image.shape[:2]
            ax3.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB), 
                      extent=[0, orig_width, 0, orig_height])
            
            # 计算缩放比例
            scale_x = orig_width / self.bounds[1].x
            scale_y = orig_height / self.bounds[1].y
            
            # 叠加处理后的地图（半透明），使用缩放后的坐标
            ax3.imshow(self.processed_map, cmap='gray', alpha=0.5, 
                      extent=[0, self.bounds[1].x * scale_x, 0, self.bounds[1].y * scale_y])
            
            # 显示障碍物（使用缩放后的坐标）
            for center, radius in self.obstacles:
                scaled_center = (center.x * scale_x, center.y * scale_y)
                scaled_radius = radius * min(scale_x, scale_y)
                circle = patches.Circle(scaled_center, scaled_radius, 
                                      color='red', alpha=0.6)
                ax3.add_patch(circle)
            
            # 添加坐标网格
            ax3.set_xlim(0, orig_width)
            ax3.set_ylim(0, orig_height)
            ax3.grid(True, alpha=0.3)
            
            # 添加一些测试点来验证坐标对应（使用缩放后的坐标）
            test_points = [
                Point(self.bounds[1].x * 0.2, self.bounds[1].y * 0.2),
                Point(self.bounds[1].x * 0.5, self.bounds[1].y * 0.5),
                Point(self.bounds[1].x * 0.8, self.bounds[1].y * 0.8)
            ]
            
            for i, point in enumerate(test_points):
                scaled_x = point.x * scale_x
                scaled_y = point.y * scale_y
                if self.is_point_in_navigable_area(point):
                    ax3.plot(scaled_x, scaled_y, 'go', markersize=8)
                    ax3.text(scaled_x, scaled_y+10, f'({point.x:.0f},{point.y:.0f})', 
                            ha='center', fontsize=8, color='green')
                else:
                    ax3.plot(scaled_x, scaled_y, 'ro', markersize=8)
                    ax3.text(scaled_x, scaled_y+10, f'({point.x:.0f},{point.y:.0f})', 
                            ha='center', fontsize=8, color='red')
            
            ax3.set_title('3. Overlay\n(Original + Processed + Test Points)')
            ax3.set_xlabel('X (World)')
            ax3.set_ylabel('Y (World)')
        else:
            ax3.text(0.5, 0.5, 'No Original Image', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('3. Overlay')
        
        plt.tight_layout()
        plt.show()
        
        # 打印坐标信息
        print(f"原始图像尺寸: {self.original_image.shape if self.original_image is not None else 'N/A'}")
        print(f"处理后地图尺寸: {self.processed_map.shape}")
        print(f"世界坐标边界: {self.bounds[0]} 到 {self.bounds[1]}")
        print(f"栅格大小: 2.0")
        print(f"栅格地图尺寸: {int(np.ceil(self.bounds[1].x / 2.0))} x {int(np.ceil(self.bounds[1].y / 2.0))}")
        
        if self.original_image is not None:
            orig_height, orig_width = self.original_image.shape[:2]
            scale_x = orig_width / self.bounds[1].x
            scale_y = orig_height / self.bounds[1].y
            print(f"缩放比例: X={scale_x:.2f}, Y={scale_y:.2f}")
            print(f"原始图像坐标范围: (0,0) 到 ({orig_width},{orig_height})")
            print(f"世界坐标范围: (0,0) 到 ({self.bounds[1].x:.0f},{self.bounds[1].y:.0f})")
        
        # 测试坐标转换
        print("\n坐标转换测试:")
        test_world_points = [
            Point(100, 100),
            Point(200, 150),
            Point(300, 200)
        ]
        
        for point in test_world_points:
            grid_x = int(point.x / 2.0)
            grid_y = int(point.y / 2.0)
            back_to_world_x = grid_x * 2.0
            back_to_world_y = grid_y * 2.0
            print(f"世界坐标: ({point.x:.1f}, {point.y:.1f}) -> 栅格坐标: ({grid_x}, {grid_y}) -> 转换回: ({back_to_world_x:.1f}, {back_to_world_y:.1f})")
    
    def interactive_point_selection(self, title: str = "Interactive Point Selection"):
        """交互式选择起点和终点"""
        print(f"\n=== {title} ===")
        print("请在地图上点击选择起点和终点")
        print("左键点击选择起点，右键点击选择终点")
        print("按 'q' 键退出选择")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 显示地图
        ax.imshow(self.processed_map, cmap='gray', extent=[0, self.bounds[1].x, self.bounds[1].y, 0])
        
        # 显示障碍物
        for center, radius in self.obstacles:
            circle = patches.Circle(center.to_tuple(), radius, 
                                  color='red', alpha=0.3)
            ax.add_patch(circle)
        
        ax.set_xlim(0, self.bounds[1].x)
        ax.set_ylim(0, self.bounds[1].y)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Click to select start (left) and goal (right) points')
        ax.grid(True, alpha=0.3)
        
        # 存储选择的点
        selected_points = {'start': None, 'goal': None}
        point_markers = {'start': None, 'goal': None}
        
        def on_click(event):
            if event.inaxes != ax:
                return
            
            if event.button == 1:  # 左键 - 起点
                point = Point(event.xdata, event.ydata)
                if self.is_point_in_navigable_area(point):
                    selected_points['start'] = point
                    # 移除旧的起点标记
                    if point_markers['start']:
                        point_markers['start'].remove()
                    # 添加新的起点标记
                    point_markers['start'] = ax.plot(point.x, point.y, 'go', markersize=12, label='Start')[0]
                    ax.text(point.x, point.y+15, 'START', ha='center', fontsize=10, color='green', weight='bold')
                    print(f"起点设置为: ({point.x:.1f}, {point.y:.1f})")
                else:
                    print(f"点击位置 ({point.x:.1f}, {point.y:.1f}) 不在可通行区域内，请重新选择")
            
            elif event.button == 3:  # 右键 - 终点
                point = Point(event.xdata, event.ydata)
                if self.is_point_in_navigable_area(point):
                    selected_points['goal'] = point
                    # 移除旧的终点标记
                    if point_markers['goal']:
                        point_markers['goal'].remove()
                    # 添加新的终点标记
                    point_markers['goal'] = ax.plot(point.x, point.y, 'ro', markersize=12, label='Goal')[0]
                    ax.text(point.x, point.y+15, 'GOAL', ha='center', fontsize=10, color='red', weight='bold')
                    print(f"终点设置为: ({point.x:.1f}, {point.y:.1f})")
                else:
                    print(f"点击位置 ({point.x:.1f}, {point.y:.1f}) 不在可通行区域内，请重新选择")
            
            # 更新图例
            if selected_points['start'] and selected_points['goal']:
                ax.legend()
            
            plt.draw()
        
        def on_key(event):
            if event.key == 'q':
                plt.close()
        
        # 绑定事件
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        plt.tight_layout()
        plt.show()
        
        return selected_points['start'], selected_points['goal']
    
    def interactive_path_planning(self):
        """交互式路径规划演示"""
        print("\n=== 交互式路径规划演示 ===")
        
        # 选择起点和终点
        start, goal = self.interactive_point_selection("选择起点和终点")
        
        if start is None or goal is None:
            print("未选择完整的起点和终点，退出演示")
            return
        
        print(f"\n选择的起点: ({start.x:.1f}, {start.y:.1f})")
        print(f"选择的终点: ({goal.x:.1f}, {goal.y:.1f})")
        
        # 选择算法
        print("\n可用的路径规划算法:")
        algorithms = ['A*', 'Dijkstra', 'RRT', 'RRT*', 'JPS']
        for i, alg in enumerate(algorithms, 1):
            print(f"{i}. {alg}")
        
        try:
            choice = input("请选择算法 (1-5): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(algorithms):
                algorithm = algorithms[int(choice) - 1]
            else:
                algorithm = 'A*'
                print(f"无效选择，使用默认算法: {algorithm}")
        except:
            algorithm = 'A*'
            print(f"使用默认算法: {algorithm}")
        
        # 规划路径
        print(f"\n使用 {algorithm} 算法规划路径...")
        path = self.plan_path(start, goal, algorithm)
        
        if path:
            length = sum(path[i].distance_to(path[i+1]) for i in range(len(path)-1))
            print(f"路径规划成功!")
            print(f"路径长度: {length:.2f}")
            print(f"路径点数: {len(path)}")
            
            # 显示路径
            self.visualize_path(start, goal, algorithm, show_map=True, show_path=True)
        else:
            print("路径规划失败!")
    
    def interactive_cleaning_planning(self):
        """交互式清扫路径规划演示"""
        print("\n=== 交互式清扫路径规划演示 ===")
        
        # 选择起点
        start, _ = self.interactive_point_selection("选择清扫起点")
        
        if start is None:
            print("未选择清扫起点，退出演示")
            return
        
        print(f"\n选择的清扫起点: ({start.x:.1f}, {start.y:.1f})")
        
        # 选择清扫算法
        print("\n可用的清扫路径算法:")
        cleaning_algorithms = ['Zigzag', 'Spiral']
        for i, alg in enumerate(cleaning_algorithms, 1):
            print(f"{i}. {alg}")
        
        try:
            choice = input("请选择清扫算法 (1-2): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(cleaning_algorithms):
                algorithm = cleaning_algorithms[int(choice) - 1]
            else:
                algorithm = 'Zigzag'
                print(f"无效选择，使用默认算法: {algorithm}")
        except:
            algorithm = 'Zigzag'
            print(f"使用默认算法: {algorithm}")
        
        # 规划清扫路径
        print(f"\n使用 {algorithm} 算法规划清扫路径...")
        path = self.plan_cleaning_path(start, algorithm)
        
        if path:
            length = sum(path[i].distance_to(path[i+1]) for i in range(len(path)-1))
            print(f"清扫路径规划成功!")
            print(f"路径长度: {length:.2f}")
            print(f"路径点数: {len(path)}")
            
            # 显示清扫路径
            self.visualize_cleaning_path(start, algorithm)
        else:
            print("清扫路径规划失败!")
    
    def visualize_map(self, show_obstacles: bool = True, show_rooms: bool = False):
        """可视化地图"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 显示原始图像
        if self.original_image is not None:
            ax.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB), 
                     extent=[0, self.bounds[1].x, self.bounds[1].y, 0])
        else:
            ax.imshow(self.processed_map, cmap='gray', 
                     extent=[0, self.bounds[1].x, self.bounds[1].y, 0])
        
        # 显示障碍物
        if show_obstacles:
            for center, radius in self.obstacles:
                circle = patches.Circle(center.to_tuple(), radius, 
                                      color='red', alpha=0.3, label='Obstacles')
                ax.add_patch(circle)
        
        # 显示房间（如果有的话）
        if show_rooms and self.rooms:
            for room in self.rooms:
                rect = patches.Rectangle(room[0].to_tuple(), room[1], room[2], 
                                       color='blue', alpha=0.2, label='Rooms')
                ax.add_patch(rect)
        
        ax.set_xlim(0, self.bounds[1].x)
        ax.set_ylim(0, self.bounds[1].y)  # 修正Y轴方向
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('户型图转机器人地图')
        ax.grid(True, alpha=0.3)
        
        if show_obstacles or show_rooms:
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def visualize_path(self, start: Point, goal: Point, algorithm: str = 'A*', 
                      show_map: bool = True, show_path: bool = True):
        """可视化路径规划结果"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 显示地图
        if show_map:
            if self.original_image is not None:
                ax.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB), 
                         extent=[0, self.bounds[1].x, self.bounds[1].y, 0])
            else:
                ax.imshow(self.processed_map, cmap='gray', 
                         extent=[0, self.bounds[1].x, self.bounds[1].y, 0])
        
        # 显示障碍物
        for center, radius in self.obstacles:
            circle = patches.Circle(center.to_tuple(), radius, 
                                  color='red', alpha=0.3)
            ax.add_patch(circle)
        
        # 显示起点和终点
        ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
        ax.plot(goal.x, goal.y, 'ro', markersize=10, label='Goal')
        
        # 显示路径
        if show_path:
            path = self.plan_path(start, goal, algorithm)
            if path:
                path_x = [p.x for p in path]
                path_y = [p.y for p in path]
                ax.plot(path_x, path_y, 'b-', linewidth=3, label=f'{algorithm} Path')
                ax.plot(path_x, path_y, 'bo', markersize=4)
            else:
                print(f"使用 {algorithm} 算法未找到路径")
        
        ax.set_xlim(0, self.bounds[1].x)
        ax.set_ylim(0, self.bounds[1].y)  # 修正Y轴方向
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'路径规划结果 - {algorithm}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_cleaning_path(self, start: Point, algorithm: str = 'Zigzag'):
        """可视化清扫路径"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 显示地图
        if self.original_image is not None:
            ax.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB), 
                     extent=[0, self.bounds[1].x, self.bounds[1].y, 0])
        else:
            ax.imshow(self.processed_map, cmap='gray', 
                     extent=[0, self.bounds[1].x, self.bounds[1].y, 0])
        
        # 显示障碍物
        for center, radius in self.obstacles:
            circle = patches.Circle(center.to_tuple(), radius, 
                                  color='red', alpha=0.3)
            ax.add_patch(circle)
        
        # 显示起点
        ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
        
        # 显示清扫路径
        path = self.plan_cleaning_path(start, algorithm)
        if path:
            path_x = [p.x for p in path]
            path_y = [p.y for p in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, label=f'{algorithm} Cleaning Path')
            ax.plot(path_x, path_y, 'bo', markersize=2, alpha=0.5)
            
            # 显示终点
            ax.plot(path[-1].x, path[-1].y, 'ro', markersize=10, label='End')
        else:
            print(f"使用 {algorithm} 算法未生成清扫路径")
        
        ax.set_xlim(0, self.bounds[1].x)
        ax.set_ylim(0, self.bounds[1].y)  # 修正Y轴方向
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'清扫路径规划 - {algorithm}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_algorithms(self, start: Point, goal: Point) -> dict:
        """比较不同算法的路径规划结果"""
        print(f"比较算法性能: 从 {start} 到 {goal}")
        print("-" * 60)
        
        results = {}
        algorithms = ['A*', 'Dijkstra', 'RRT', 'RRT*', 'JPS']
        
        for algorithm in algorithms:
            import time
            start_time = time.time()
            path = self.plan_path(start, goal, algorithm)
            end_time = time.time()
            
            if path:
                length = sum(path[i].distance_to(path[i+1]) for i in range(len(path)-1))
                results[algorithm] = {
                    'path': path,
                    'length': length,
                    'points': len(path),
                    'time': end_time - start_time
                }
                print(f"{algorithm:<10} 长度: {length:<8.2f} 点数: {len(path):<6} 时间: {(end_time-start_time)*1000:.1f}ms")
            else:
                results[algorithm] = {
                    'path': [],
                    'length': float('inf'),
                    'points': 0,
                    'time': end_time - start_time
                }
                print(f"{algorithm:<10} 未找到路径 时间: {(end_time-start_time)*1000:.1f}ms")
        
        return results
    
    def save_path_to_file(self, path: List[Point], filename: str):
        """保存路径到文件"""
        with open(filename, 'w') as f:
            f.write("# 路径点坐标 (x, y)\n")
            for i, point in enumerate(path):
                f.write(f"{i+1},{point.x:.2f},{point.y:.2f}\n")
        print(f"路径已保存到: {filename}")

def demo_with_existing_images():
    """使用现有图像进行演示"""
    print("=== 户型图转机器人地图演示 ===")
    
    # 查找现有的户型图
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        import glob
        image_files.extend(glob.glob(f"/media/lixiang/matrix/Documents/code/opensource/floor_plane/{ext}"))
    
    if not image_files:
        print("未找到户型图文件")
        return
    
    print(f"找到图像文件: {image_files}")
    
    # 使用第一个图像
    image_path = image_files[0]
    print(f"使用图像: {image_path}")
    
    try:
        # 创建系统
        system = HousePlanToRobotMap(image_path, scale_factor=0.5)
        
        # 显示地图
        system.visualize_map()
        
        # 设置起点和终点
        bounds = system.bounds
        start = Point(bounds[0].x + 10, bounds[0].y + 10)
        goal = Point(bounds[1].x - 10, bounds[1].y - 10)
        
        print(f"起点: {start}")
        print(f"终点: {goal}")
        
        # 比较算法
        results = system.compare_algorithms(start, goal)
        
        # 可视化最佳路径
        best_algorithm = min(results.keys(), key=lambda k: results[k]['length'])
        print(f"\n最佳算法: {best_algorithm}")
        system.visualize_path(start, goal, best_algorithm)
        
        # 演示清扫路径
        print("\n=== 清扫路径演示 ===")
        system.visualize_cleaning_path(start, 'Zigzag')
        system.visualize_cleaning_path(start, 'Spiral')
        
    except Exception as e:
        print(f"错误: {e}")

def interactive_demo():
    """交互式演示"""
    print("=== 交互式户型图转机器人地图演示 ===")
    
    # 查找图像文件
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        import glob
        image_files.extend(glob.glob(f"/media/lixiang/matrix/Documents/code/opensource/floor_plane/{ext}"))
    
    if not image_files:
        print("未找到户型图文件")
        return
    
    print("可用的图像文件:")
    for i, img in enumerate(image_files):
        print(f"{i+1}. {img}")
    
    try:
        choice = int(input("请选择图像文件 (输入数字): ")) - 1
        if choice < 0 or choice >= len(image_files):
            print("无效选择")
            return
        
        image_path = image_files[choice]
        print(f"选择图像: {image_path}")
        
        # 创建系统
        system = HousePlanToRobotMap(image_path, scale_factor=0.5)
        
        while True:
            print("\n请选择操作:")
            print("1. 显示地图")
            print("2. 路径规划")
            print("3. 清扫路径规划")
            print("4. 比较算法")
            print("5. 退出")
            
            choice = input("请输入选择 (1-5): ").strip()
            
            if choice == '1':
                system.visualize_map()
            
            elif choice == '2':
                try:
                    start_x = float(input("起点X坐标: "))
                    start_y = float(input("起点Y坐标: "))
                    goal_x = float(input("终点X坐标: "))
                    goal_y = float(input("终点Y坐标: "))
                    
                    start = Point(start_x, start_y)
                    goal = Point(goal_x, goal_y)
                    
                    algorithm = input("算法 (A*, Dijkstra, RRT, RRT*, JPS): ").strip() or 'A*'
                    system.visualize_path(start, goal, algorithm)
                    
                except ValueError:
                    print("输入格式错误")
            
            elif choice == '3':
                try:
                    start_x = float(input("起点X坐标: "))
                    start_y = float(input("起点Y坐标: "))
                    start = Point(start_x, start_y)
                    
                    algorithm = input("清扫算法 (Zigzag, Spiral): ").strip() or 'Zigzag'
                    system.visualize_cleaning_path(start, algorithm)
                    
                except ValueError:
                    print("输入格式错误")
            
            elif choice == '4':
                try:
                    start_x = float(input("起点X坐标: "))
                    start_y = float(input("起点Y坐标: "))
                    goal_x = float(input("终点X坐标: "))
                    goal_y = float(input("终点Y坐标: "))
                    
                    start = Point(start_x, start_y)
                    goal = Point(goal_x, goal_y)
                    
                    results = system.compare_algorithms(start, goal)
                    
                except ValueError:
                    print("输入格式错误")
            
            elif choice == '5':
                break
            
            else:
                print("无效选择")
    
    except KeyboardInterrupt:
        print("\n演示结束")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    print("户型图转机器人地图和路径规划系统")
    print("=" * 50)
    
    # 运行演示
    demo_with_existing_images()
    
    # 交互式演示
    # interactive_demo()
