#!/usr/bin/env python3

"""
综合测试文件 - 户型图转机器人地图系统
整合了所有测试功能：
1. 基础功能演示
2. 路径规划算法测试
3. 交互式演示
4. 性能比较
5. 清扫路径演示
"""

import sys
import os
import glob
import time
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'script'))

from house_plan_to_robot_map import HousePlanToRobotMap
from floor_plan.path_planning import *

class ComprehensiveTest:
    """综合测试类"""
    
    def __init__(self):
        self.converter = None
        self.image_files = []
        self.find_image_files()
    
    def find_image_files(self):
        """查找图像文件"""
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_files.extend(glob.glob(f"/media/lixiang/matrix/Documents/code/opensource/floor_plane/{ext}"))
        
        if not self.image_files:
            print("未找到户型图文件")
            print("请将户型图文件放在项目根目录下")
            return
        
        print(f"找到图像文件: {self.image_files}")
    
    def load_map(self, image_path=None):
        """加载地图"""
        if not self.image_files:
            return False
        
        if image_path is None:
            image_path = self.image_files[0]
        
        print(f"使用图像: {os.path.basename(image_path)}")
        print("正在处理户型图...")
        
        self.converter = HousePlanToRobotMap(image_path, obstacle_scale=0.3)
        return True
    
    def test_basic_functionality(self):
        """测试基础功能"""
        print("\n" + "="*60)
        print("=== 基础功能测试 ===")
        print("="*60)
        
        if not self.load_map():
            return
        
        # 显示地图转换过程
        self.converter.visualize_map_conversion()
        
        # 显示最终地图
        self.converter.visualize_map()
        
        print("基础功能测试完成！")
    
    def test_path_planning_algorithms(self):
        """测试路径规划算法"""
        print("\n" + "="*60)
        print("=== 路径规划算法测试 ===")
        print("="*60)
        
        if not self.converter:
            if not self.load_map():
                return
        
        # 设置测试点
        start = Point(87.4, 60.0)
        goal = Point(349.6, 240.0)
        
        print(f"起点: ({start.x}, {start.y})")
        print(f"终点: ({goal.x}, {goal.y})")
        
        # 测试各种算法
        algorithms = ['A*', 'Dijkstra', 'RRT', 'RRT*']
        
        for algo in algorithms:
            print(f"\n--- {algo}算法 ---")
            try:
                path = self.converter.plan_path(start, goal, algo)
                if path:
                    length = sum(path[i].distance_to(path[i+1]) for i in range(len(path)-1))
                    print(f"{algo}路径长度: {length:.2f}, 点数: {len(path)}")
                else:
                    print(f"{algo}未找到路径")
            except Exception as e:
                print(f"{algo}算法错误: {e}")
        
        # 可视化A*路径
        if 'A*' in self.converter.planners:
            path = self.converter.plan_path(start, goal, 'A*')
            if path:
                self.converter.visualize_path(start, goal, 'A*')
    
    def test_cleaning_algorithms(self):
        """测试清扫算法"""
        print("\n" + "="*60)
        print("=== 清扫算法测试 ===")
        print("="*60)
        
        if not self.converter:
            if not self.load_map():
                return
        
        start = Point(30.0, 30.0)
        print(f"清扫起点: ({start.x}, {start.y})")
        
        # 测试Zigzag算法
        print("\n--- Zigzag清扫路径 ---")
        zigzag_path = self.converter.plan_cleaning_path(start, 'Zigzag')
        if zigzag_path:
            length = sum(zigzag_path[i].distance_to(zigzag_path[i+1]) for i in range(len(zigzag_path)-1))
            print(f"Zigzag路径长度: {length:.2f}, 点数: {len(zigzag_path)}")
        
        # 测试Spiral算法
        print("\n--- Spiral清扫路径 ---")
        spiral_path = self.converter.plan_cleaning_path(start, 'Spiral')
        if spiral_path:
            length = sum(spiral_path[i].distance_to(spiral_path[i+1]) for i in range(len(spiral_path)-1))
            print(f"Spiral路径长度: {length:.2f}, 点数: {len(spiral_path)}")
        
        # 可视化清扫路径
        if zigzag_path:
            self.converter.visualize_cleaning_path(start, "Zigzag")
    
    def test_algorithm_performance(self):
        """测试算法性能"""
        print("\n" + "="*60)
        print("=== 算法性能比较 ===")
        print("="*60)
        
        if not self.converter:
            if not self.load_map():
                return
        
        start = Point(87.4, 60.0)
        goal = Point(349.6, 240.0)
        
        print(f"比较算法性能: 从 {start} 到 {goal}")
        print("-" * 60)
        
        algorithms = ['A*', 'Dijkstra', 'RRT', 'RRT*']
        results = []
        
        for algo in algorithms:
            try:
                start_time = time.time()
                path = self.converter.plan_path(start, goal, algo)
                end_time = time.time()
                
                if path:
                    length = sum(path[i].distance_to(path[i+1]) for i in range(len(path)-1))
                    time_ms = (end_time - start_time) * 1000
                    results.append((algo, length, len(path), time_ms))
                    print(f"{algo:<10} 长度: {length:8.2f}   点数: {len(path):4d}    时间: {time_ms:6.1f}ms")
                else:
                    print(f"{algo:<10} 未找到路径")
            except Exception as e:
                print(f"{algo:<10} 错误: {e}")
        
        return results
    
    def test_wall_collision(self):
        """测试墙壁碰撞检测"""
        print("\n" + "="*60)
        print("=== 墙壁碰撞检测测试 ===")
        print("="*60)
        
        if not self.converter:
            if not self.load_map():
                return
        
        # 测试一些点
        test_points = [
            Point(50, 50),    # 应该在墙壁上
            Point(100, 100),  # 应该在可通行区域
            Point(200, 150),  # 应该在墙壁上
            Point(300, 200),  # 应该在可通行区域
            Point(50, 200),   # 应该在墙壁上
        ]
        
        print("测试点检查结果:")
        for point in test_points:
            is_navigable = self.converter.is_point_in_navigable_area(point)
            is_collision = self.converter.planners['A*'].is_collision(point)
            print(f"点 ({point.x}, {point.y}): 可通行={is_navigable}, 碰撞={is_collision}")
    
    def interactive_demo(self):
        """交互式演示"""
        print("\n" + "="*60)
        print("=== 交互式演示 ===")
        print("="*60)
        
        if not self.converter:
            if not self.load_map():
                return
        
        print("启动交互式路径规划...")
        print("点击地图选择起点和终点")
        
        try:
            self.converter.interactive_path_planning()
        except Exception as e:
            print(f"交互式演示错误: {e}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("户型图转机器人地图系统 - 综合测试")
        print("="*60)
        
        # 1. 基础功能测试
        self.test_basic_functionality()
        
        # 2. 墙壁碰撞检测测试
        self.test_wall_collision()
        
        # 3. 路径规划算法测试
        self.test_path_planning_algorithms()
        
        # 4. 清扫算法测试
        self.test_cleaning_algorithms()
        
        # 5. 算法性能比较
        self.test_algorithm_performance()
        
        print("\n" + "="*60)
        print("=== 所有测试完成 ===")
        print("="*60)
        print("要运行交互式演示，请调用: test.interactive_demo()")
    
    def run_interactive_only(self):
        """只运行交互式演示"""
        if not self.load_map():
            return
        
        self.interactive_demo()

def main():
    """主函数"""
    test = ComprehensiveTest()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        # 只运行交互式演示
        test.run_interactive_only()
    else:
        # 运行所有测试
        test.run_all_tests()

if __name__ == "__main__":
    main()
