#! /usr/bin/env python3

import numpy as np
import heapq
import math
import random
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.patches as patches

@dataclass
class Point:
    """2D点类"""
    x: float
    y: float
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar):
        return Point(self.x / scalar, self.y / scalar)
    
    def distance_to(self, other) -> float:
        """计算到另一个点的欧几里得距离"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

@dataclass
class Node:
    """路径规划中的节点"""
    point: Point
    g_cost: float = 0.0  # 从起点到当前节点的实际代价
    h_cost: float = 0.0  # 从当前节点到终点的启发式代价
    f_cost: float = 0.0  # 总代价 f = g + h
    parent: Optional['Node'] = None
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost

class PathPlanner(ABC):
    """路径规划算法基类"""
    
    def __init__(self, grid_size: float = 1.0):
        self.grid_size = grid_size
        self.obstacles: List[Tuple[Point, float]] = []  # (center, radius) 圆形障碍物
        self.obstacle_map: Optional[np.ndarray] = None  # 栅格地图
        self.wall_checker = None  # 墙壁检查函数
        
    def add_obstacle(self, center: Point, radius: float):
        """添加圆形障碍物"""
        self.obstacles.append((center, radius))
    
    def add_rectangular_obstacle(self, top_left: Point, width: float, height: float):
        """添加矩形障碍物（转换为多个圆形障碍物近似）"""
        # 将矩形分解为多个小圆形
        step = self.grid_size / 2
        for x in np.arange(top_left.x, top_left.x + width, step):
            for y in np.arange(top_left.y, top_left.y + height, step):
                self.add_obstacle(Point(x, y), step)
    
    def is_collision(self, point: Point) -> bool:
        """检查点是否与障碍物碰撞"""
        # 检查圆形障碍物
        for center, radius in self.obstacles:
            if point.distance_to(center) <= radius:
                return True
        
        # 检查墙壁（如果提供了墙壁检查函数）
        if self.wall_checker is not None:
            if not self.wall_checker(point):
                return True
                
        return False
    
    def is_valid_point(self, point: Point, bounds: Tuple[Point, Point]) -> bool:
        """检查点是否在边界内且无碰撞"""
        min_bound, max_bound = bounds
        if not (min_bound.x <= point.x <= max_bound.x and 
                min_bound.y <= point.y <= max_bound.y):
            return False
        return not self.is_collision(point)
    
    @abstractmethod
    def plan(self, start: Point, goal: Point, bounds: Tuple[Point, Point]) -> List[Point]:
        """规划路径，返回路径点列表"""
        pass
    
    def smooth_path(self, path: List[Point], bounds: Tuple[Point, Point]) -> List[Point]:
        """路径平滑化"""
        if len(path) < 3:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            j = len(path) - 1
            # 从最远的点开始，检查是否可以直接连接
            while j > i + 1:
                if self._is_line_clear(path[i], path[j], bounds):
                    smoothed.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                # 如果无法跳过任何点，则移动到下一个点
                i += 1
                if i < len(path):
                    smoothed.append(path[i])
        
        return smoothed
    
    def _is_line_clear(self, start: Point, end: Point, bounds: Tuple[Point, Point]) -> bool:
        """检查两点间的直线是否无碰撞"""
        steps = int(start.distance_to(end) / (self.grid_size / 4)) + 1
        for i in range(steps + 1):
            t = i / steps
            point = Point(
                start.x + t * (end.x - start.x),
                start.y + t * (end.y - start.y)
            )
            if not self.is_valid_point(point, bounds):
                return False
        return True

class AStar(PathPlanner):
    """A*算法实现"""
    
    def __init__(self, grid_size: float = 1.0, diagonal_movement: bool = True):
        super().__init__(grid_size)
        self.diagonal_movement = diagonal_movement
        
    def plan(self, start: Point, goal: Point, bounds: Tuple[Point, Point]) -> List[Point]:
        """A*路径规划"""
        if not self.is_valid_point(start, bounds) or not self.is_valid_point(goal, bounds):
            return []
        
        # 将连续空间离散化为栅格
        min_bound, max_bound = bounds
        # 确保栅格能完全覆盖地图范围
        width = int(np.ceil((max_bound.x - min_bound.x) / self.grid_size))
        height = int(np.ceil((max_bound.y - min_bound.y) / self.grid_size))
        
        # 创建栅格地图
        grid = np.zeros((height, width), dtype=bool)
        for i in range(height):
            for j in range(width):
                point = Point(
                    min_bound.x + j * self.grid_size,
                    min_bound.y + i * self.grid_size
                )
                # 检查圆形障碍物
                collision = False
                for center, radius in self.obstacles:
                    if point.distance_to(center) <= radius:
                        collision = True
                        break
                
                # 检查墙壁（如果提供了墙壁检查函数）
                if not collision and self.wall_checker is not None:
                    if not self.wall_checker(point):
                        collision = True
                
                grid[i, j] = collision
        
        # 起点和终点在栅格中的位置
        start_grid = (
            int((start.x - min_bound.x) / self.grid_size),
            int((start.y - min_bound.y) / self.grid_size)
        )
        goal_grid = (
            int((goal.x - min_bound.x) / self.grid_size),
            int((goal.y - min_bound.y) / self.grid_size)
        )
        
        # A*算法
        open_set = []
        closed_set = set()
        came_from = {}
        
        # A*算法 - 使用栅格坐标进行搜索
        open_set = []
        closed_set = set()
        came_from = {}
        g_score = {}
        f_score = {}
        
        # 使用栅格坐标作为节点标识
        start_pos = start_grid
        goal_pos = goal_grid
        
        g_score[start_pos] = 0
        f_score[start_pos] = self._heuristic(start_pos, goal_pos)
        
        heapq.heappush(open_set, (f_score[start_pos], start_pos))
        
        search_count = 0
        max_iterations = 10000
        
        while open_set and search_count < max_iterations:
            search_count += 1
            
            current = heapq.heappop(open_set)[1]
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            if current == goal_pos:
                # 重构路径
                path = []
                while current in came_from:
                    world_point = Point(
                        min_bound.x + current[0] * self.grid_size,
                        min_bound.y + current[1] * self.grid_size
                    )
                    path.append(world_point)
                    current = came_from[current]
                
                # 添加起点
                world_point = Point(
                    min_bound.x + start_pos[0] * self.grid_size,
                    min_bound.y + start_pos[1] * self.grid_size
                )
                path.append(world_point)
                
                return list(reversed(path))
            
            # 检查邻居
            for dx, dy in self._get_neighbors():
                neighbor_x = current[0] + dx
                neighbor_y = current[1] + dy
                
                if (neighbor_x < 0 or neighbor_x >= width or 
                    neighbor_y < 0 or neighbor_y >= height):
                    continue
                
                if grid[neighbor_y, neighbor_x]:
                    continue
                
                if (neighbor_x, neighbor_y) in closed_set:
                    continue
                
                # 计算移动代价
                move_cost = math.sqrt(dx*dx + dy*dy) * self.grid_size
                tentative_g = g_score[current] + move_cost
                
                neighbor_pos = (neighbor_x, neighbor_y)
                
                # 检查是否已经在开放集中
                if neighbor_pos not in g_score or tentative_g < g_score[neighbor_pos]:
                    came_from[neighbor_pos] = current
                    g_score[neighbor_pos] = tentative_g
                    f_score[neighbor_pos] = tentative_g + self._heuristic(neighbor_pos, goal_pos)
                    
                    # 检查是否已经在开放集中
                    in_open = False
                    for _, pos in open_set:
                        if pos == neighbor_pos:
                            in_open = True
                            break
                    
                    if not in_open:
                        heapq.heappush(open_set, (f_score[neighbor_pos], neighbor_pos))
        
        return []  # 未找到路径
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """启发式函数（欧几里得距离）"""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        if self.diagonal_movement:
            return math.sqrt(dx*dx + dy*dy) * self.grid_size
        else:
            return (dx + dy) * self.grid_size
    
    def _get_neighbors(self) -> List[Tuple[int, int]]:
        """获取邻居方向"""
        if self.diagonal_movement:
            return [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            return [(-1, 0), (1, 0), (0, -1), (0, 1)]

class Dijkstra(PathPlanner):
    """Dijkstra算法实现"""
    
    def plan(self, start: Point, goal: Point, bounds: Tuple[Point, Point]) -> List[Point]:
        """Dijkstra路径规划"""
        if not self.is_valid_point(start, bounds) or not self.is_valid_point(goal, bounds):
            return []
        
        # 将连续空间离散化为栅格
        min_bound, max_bound = bounds
        # 确保栅格能完全覆盖地图范围
        width = int(np.ceil((max_bound.x - min_bound.x) / self.grid_size))
        height = int(np.ceil((max_bound.y - min_bound.y) / self.grid_size))
        
        # 创建栅格地图
        grid = np.zeros((height, width), dtype=bool)
        for i in range(height):
            for j in range(width):
                point = Point(
                    min_bound.x + j * self.grid_size,
                    min_bound.y + i * self.grid_size
                )
                grid[i, j] = self.is_collision(point)
        
        # 起点和终点在栅格中的位置
        start_grid = (
            int((start.x - min_bound.x) / self.grid_size),
            int((start.y - min_bound.y) / self.grid_size)
        )
        goal_grid = (
            int((goal.x - min_bound.x) / self.grid_size),
            int((goal.y - min_bound.y) / self.grid_size)
        )
        
        # Dijkstra算法
        distances = np.full((height, width), float('inf'))
        distances[start_grid[1], start_grid[0]] = 0
        came_from = {}
        
        open_set = [(0, start_grid[0], start_grid[1])]
        closed_set = set()
        
        while open_set:
            current_dist, current_x, current_y = heapq.heappop(open_set)
            current_pos = (current_x, current_y)
            
            if current_pos in closed_set:
                continue
                
            closed_set.add(current_pos)
            
            if current_pos == goal_grid:
                # 重构路径
                path = []
                while current_pos in came_from:
                    world_point = Point(
                        min_bound.x + current_pos[0] * self.grid_size,
                        min_bound.y + current_pos[1] * self.grid_size
                    )
                    path.append(world_point)
                    current_pos = came_from[current_pos]
                
                world_point = Point(
                    min_bound.x + start_grid[0] * self.grid_size,
                    min_bound.y + start_grid[1] * self.grid_size
                )
                path.append(world_point)
                return list(reversed(path))
            
            # 检查邻居
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor_x = current_x + dx
                neighbor_y = current_y + dy
                
                if (neighbor_x < 0 or neighbor_x >= width or 
                    neighbor_y < 0 or neighbor_y >= height):
                    continue
                
                if grid[neighbor_y, neighbor_x]:
                    continue
                
                if (neighbor_x, neighbor_y) in closed_set:
                    continue
                
                move_cost = self.grid_size
                tentative_dist = current_dist + move_cost
                
                if tentative_dist < distances[neighbor_y, neighbor_x]:
                    distances[neighbor_y, neighbor_x] = tentative_dist
                    came_from[(neighbor_x, neighbor_y)] = current_pos
                    heapq.heappush(open_set, (tentative_dist, neighbor_x, neighbor_y))
        
        return []  # 未找到路径

class RRT(PathPlanner):
    """RRT (Rapidly-exploring Random Tree) 算法实现"""
    
    def __init__(self, grid_size: float = 1.0, step_size: float = 1.0, max_iterations: int = 1000):
        super().__init__(grid_size)
        self.step_size = step_size
        self.max_iterations = max_iterations
    
    def plan(self, start: Point, goal: Point, bounds: Tuple[Point, Point]) -> List[Point]:
        """RRT路径规划"""
        if not self.is_valid_point(start, bounds) or not self.is_valid_point(goal, bounds):
            return []
        
        # 初始化树
        tree = {start.to_tuple(): None}  # 节点到父节点的映射
        nodes = [start]
        
        for _ in range(self.max_iterations):
            # 生成随机点
            if random.random() < 0.1:  # 10%概率选择目标点
                random_point = goal
            else:
                random_point = self._random_point(bounds)
            
            # 找到最近的树节点
            nearest_node = min(nodes, key=lambda n: n.distance_to(random_point))
            
            # 向随机点方向扩展
            direction = Point(
                random_point.x - nearest_node.x,
                random_point.y - nearest_node.y
            )
            distance = nearest_node.distance_to(random_point)
            
            if distance > 0:
                direction = Point(
                    direction.x / distance * min(self.step_size, distance),
                    direction.y / distance * min(self.step_size, distance)
                )
            
            new_point = Point(nearest_node.x + direction.x, nearest_node.y + direction.y)
            
            # 检查路径是否有效
            if self._is_line_clear(nearest_node, new_point, bounds):
                tree[new_point.to_tuple()] = nearest_node
                nodes.append(new_point)
                
                # 检查是否到达目标
                if new_point.distance_to(goal) < self.step_size:
                    if self._is_line_clear(new_point, goal, bounds):
                        tree[goal.to_tuple()] = new_point
                        nodes.append(goal)
                        break
        
        # 重构路径
        if goal.to_tuple() not in tree:
            return []
        
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = tree[current.to_tuple()]
        
        return list(reversed(path))
    
    def _random_point(self, bounds: Tuple[Point, Point]) -> Point:
        """生成随机点"""
        min_bound, max_bound = bounds
        return Point(
            random.uniform(min_bound.x, max_bound.x),
            random.uniform(min_bound.y, max_bound.y)
        )

class RRTStar(RRT):
    """RRT*算法实现（RRT的优化版本）"""
    
    def __init__(self, grid_size: float = 1.0, step_size: float = 1.0, 
                 max_iterations: int = 1000, rewire_radius: float = 2.0):
        super().__init__(grid_size, step_size, max_iterations)
        self.rewire_radius = rewire_radius
    
    def plan(self, start: Point, goal: Point, bounds: Tuple[Point, Point]) -> List[Point]:
        """RRT*路径规划"""
        if not self.is_valid_point(start, bounds) or not self.is_valid_point(goal, bounds):
            return []
        
        # 初始化树
        tree = {start.to_tuple(): None}  # 节点到父节点的映射
        costs = {start.to_tuple(): 0.0}  # 节点到起点的代价
        nodes = [start]
        
        for _ in range(self.max_iterations):
            # 生成随机点
            if random.random() < 0.1:  # 10%概率选择目标点
                random_point = goal
            else:
                random_point = self._random_point(bounds)
            
            # 找到最近的树节点
            nearest_node = min(nodes, key=lambda n: n.distance_to(random_point))
            
            # 向随机点方向扩展
            direction = Point(
                random_point.x - nearest_node.x,
                random_point.y - nearest_node.y
            )
            distance = nearest_node.distance_to(random_point)
            
            if distance > 0:
                direction = Point(
                    direction.x / distance * min(self.step_size, distance),
                    direction.y / distance * min(self.step_size, distance)
                )
            
            new_point = Point(nearest_node.x + direction.x, nearest_node.y + direction.y)
            
            # 检查路径是否有效
            if self._is_line_clear(nearest_node, new_point, bounds):
                # 找到新节点的最佳父节点
                best_parent = nearest_node
                best_cost = costs[nearest_node] + nearest_node.distance_to(new_point)
                
                # 在重连半径内寻找更好的父节点
                for node in nodes:
                    if (node.distance_to(new_point) <= self.rewire_radius and
                        self._is_line_clear(node, new_point, bounds)):
                        cost = costs[node] + node.distance_to(new_point)
                        if cost < best_cost:
                            best_parent = node
                            best_cost = cost
                
                tree[new_point] = best_parent
                costs[new_point] = best_cost
                nodes.append(new_point)
                
                # 重连：尝试改善附近节点的路径
                for node in nodes:
                    if (node != new_point and 
                        node.distance_to(new_point) <= self.rewire_radius and
                        self._is_line_clear(new_point, node, bounds)):
                        new_cost = costs[new_point] + new_point.distance_to(node)
                        if new_cost < costs[node]:
                            tree[node] = new_point
                            costs[node] = new_cost
                
                # 检查是否到达目标
                if new_point.distance_to(goal) < self.step_size:
                    if self._is_line_clear(new_point, goal, bounds):
                        goal_cost = costs[new_point] + new_point.distance_to(goal)
                        if goal not in costs or goal_cost < costs[goal]:
                            tree[goal] = new_point
                            costs[goal] = goal_cost
                            nodes.append(goal)
        
        # 重构路径
        if goal not in tree:
            return []
        
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = tree[current]
        
        return list(reversed(path))

class JPS(PathPlanner):
    """JPS (Jump Point Search) 算法实现"""
    
    def plan(self, start: Point, goal: Point, bounds: Tuple[Point, Point]) -> List[Point]:
        """JPS路径规划"""
        if not self.is_valid_point(start, bounds) or not self.is_valid_point(goal, bounds):
            return []
        
        # 将连续空间离散化为栅格
        min_bound, max_bound = bounds
        # 确保栅格能完全覆盖地图范围
        width = int(np.ceil((max_bound.x - min_bound.x) / self.grid_size))
        height = int(np.ceil((max_bound.y - min_bound.y) / self.grid_size))
        
        # 创建栅格地图
        grid = np.zeros((height, width), dtype=bool)
        for i in range(height):
            for j in range(width):
                point = Point(
                    min_bound.x + j * self.grid_size,
                    min_bound.y + i * self.grid_size
                )
                grid[i, j] = self.is_collision(point)
        
        # 起点和终点在栅格中的位置
        start_grid = (
            int((start.x - min_bound.x) / self.grid_size),
            int((start.y - min_bound.y) / self.grid_size)
        )
        goal_grid = (
            int((goal.x - min_bound.x) / self.grid_size),
            int((goal.y - min_bound.y) / self.grid_size)
        )
        
        # JPS算法
        open_set = []
        closed_set = set()
        came_from = {}
        g_scores = {}
        
        start_node = Node(Point(start_grid[0], start_grid[1]))
        start_node.g_cost = 0
        start_node.h_cost = self._heuristic(start_grid, goal_grid)
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        
        heapq.heappush(open_set, start_node)
        g_scores[start_grid] = 0
        
        while open_set:
            current = heapq.heappop(open_set)
            current_pos = (int(current.point.x), int(current.point.y))
            
            if current_pos in closed_set:
                continue
                
            closed_set.add(current_pos)
            
            if current_pos == goal_grid:
                # 重构路径
                path = []
                while current:
                    world_point = Point(
                        min_bound.x + current.point.x * self.grid_size,
                        min_bound.y + current.point.y * self.grid_size
                    )
                    path.append(world_point)
                    current = current.parent
                return list(reversed(path))
            
            # 获取跳跃点
            jump_points = self._get_jump_points(current_pos, goal_grid, grid, width, height)
            
            for jump_point in jump_points:
                if jump_point in closed_set:
                    continue
                
                # 计算代价
                move_cost = math.sqrt(
                    (jump_point[0] - current_pos[0])**2 + 
                    (jump_point[1] - current_pos[1])**2
                ) * self.grid_size
                
                tentative_g = g_scores[current_pos] + move_cost
                
                if jump_point not in g_scores or tentative_g < g_scores[jump_point]:
                    g_scores[jump_point] = tentative_g
                    
                    jump_node = Node(Point(jump_point[0], jump_point[1]))
                    jump_node.g_cost = tentative_g
                    jump_node.h_cost = self._heuristic(jump_point, goal_grid)
                    jump_node.f_cost = jump_node.g_cost + jump_node.h_cost
                    jump_node.parent = current
                    
                    heapq.heappush(open_set, jump_node)
        
        return []  # 未找到路径
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """启发式函数（对角线距离）"""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        return (dx + dy + (math.sqrt(2) - 2) * min(dx, dy)) * self.grid_size
    
    def _get_jump_points(self, current: Tuple[int, int], goal: Tuple[int, int], 
                        grid: np.ndarray, width: int, height: int) -> List[Tuple[int, int]]:
        """获取跳跃点"""
        jump_points = []
        
        # 8个方向
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dx, dy in directions:
            jump_point = self._jump(current, dx, dy, goal, grid, width, height)
            if jump_point:
                jump_points.append(jump_point)
        
        return jump_points
    
    def _jump(self, current: Tuple[int, int], dx: int, dy: int, goal: Tuple[int, int],
              grid: np.ndarray, width: int, height: int) -> Optional[Tuple[int, int]]:
        """跳跃函数"""
        next_x = current[0] + dx
        next_y = current[1] + dy
        
        # 检查边界
        if (next_x < 0 or next_x >= width or next_y < 0 or next_y >= height):
            return None
        
        # 检查障碍物
        if grid[next_y, next_x]:
            return None
        
        # 如果到达目标
        if (next_x, next_y) == goal:
            return (next_x, next_y)
        
        # 检查是否遇到障碍物（强制邻居）
        if dx != 0 and dy != 0:  # 对角线移动
            if (grid[current[1] + dy, current[0]] or 
                grid[current[1], current[0] + dx]):
                return (next_x, next_y)
            
            # 递归跳跃
            horizontal_jump = self._jump((next_x, next_y), dx, 0, goal, grid, width, height)
            vertical_jump = self._jump((next_x, next_y), 0, dy, goal, grid, width, height)
            
            if horizontal_jump or vertical_jump:
                return (next_x, next_y)
        else:  # 水平或垂直移动
            if dx != 0:  # 水平移动
                if ((next_y > 0 and grid[next_y - 1, next_x] and 
                     not grid[next_y - 1, next_x - dx]) or
                    (next_y < height - 1 and grid[next_y + 1, next_x] and 
                     not grid[next_y + 1, next_x - dx])):
                    return (next_x, next_y)
            else:  # 垂直移动
                if ((next_x > 0 and grid[next_y, next_x - 1] and 
                     not grid[next_y - dy, next_x - 1]) or
                    (next_x < width - 1 and grid[next_y, next_x + 1] and 
                     not grid[next_y - dy, next_x + 1])):
                    return (next_x, next_y)
        
        # 继续跳跃
        return self._jump((next_x, next_y), dx, dy, goal, grid, width, height)

class ZigzagPlanner(PathPlanner):
    """Zigzag路径规划器 - 专为扫地机器人设计"""
    
    def __init__(self, grid_size: float = 1.0, sweep_width: float = 0.5):
        super().__init__(grid_size)
        self.sweep_width = sweep_width
    
    def plan(self, start: Point, goal: Point, bounds: Tuple[Point, Point]) -> List[Point]:
        """生成Zigzag清扫路径"""
        min_bound, max_bound = bounds
        
        # 计算清扫区域
        sweep_area = self._calculate_sweep_area(min_bound, max_bound)
        if not sweep_area:
            return []
        
        # 生成zigzag路径
        zigzag_path = self._generate_zigzag_path(sweep_area, min_bound, max_bound)
        
        # 从起点连接到清扫路径
        if zigzag_path:
            # 找到最近的清扫路径点
            nearest_point = min(zigzag_path, key=lambda p: start.distance_to(p))
            nearest_idx = zigzag_path.index(nearest_point)
            
            # 从起点到最近点
            start_path = self._connect_points(start, nearest_point, bounds)
            if start_path:
                zigzag_path = start_path[:-1] + zigzag_path[nearest_idx:]
            
            # 从清扫路径到终点
            end_path = self._connect_points(zigzag_path[-1], goal, bounds)
            if end_path:
                zigzag_path.extend(end_path[1:])
        
        return zigzag_path
    
    def _calculate_sweep_area(self, min_bound: Point, max_bound: Point) -> List[Point]:
        """计算需要清扫的区域"""
        # 简化版本：返回整个区域的关键点
        # 实际应用中可以根据障碍物和房间布局来计算
        area_points = []
        
        # 生成网格点
        x_step = self.sweep_width
        y_step = self.sweep_width
        
        x = min_bound.x + x_step / 2
        while x < max_bound.x:
            y = min_bound.y + y_step / 2
            while y < max_bound.y:
                point = Point(x, y)
                if not self.is_collision(point):
                    area_points.append(point)
                y += y_step
            x += x_step
        
        return area_points
    
    def _generate_zigzag_path(self, area_points: List[Point], 
                             min_bound: Point, max_bound: Point) -> List[Point]:
        """生成zigzag清扫路径"""
        if not area_points:
            return []
        
        # 按Y坐标分组
        rows = {}
        for point in area_points:
            y_key = round(point.y / self.sweep_width) * self.sweep_width
            if y_key not in rows:
                rows[y_key] = []
            rows[y_key].append(point)
        
        # 按Y坐标排序
        sorted_rows = sorted(rows.items())
        
        path = []
        left_to_right = True
        
        for y_key, points in sorted_rows:
            # 按X坐标排序
            points.sort(key=lambda p: p.x)
            
            if left_to_right:
                path.extend(points)
            else:
                path.extend(reversed(points))
            
            left_to_right = not left_to_right
        
        return path
    
    def _connect_points(self, start: Point, end: Point, bounds: Tuple[Point, Point]) -> List[Point]:
        """连接两个点，避开障碍物"""
        # 简单的直线连接，如果被阻挡则使用A*算法
        if self._is_line_clear(start, end, bounds):
            return [start, end]
        else:
            # 使用A*算法连接
            astar = AStar(self.grid_size)
            astar.obstacles = self.obstacles.copy()
            return astar.plan(start, end, bounds)

class SpiralPlanner(PathPlanner):
    """螺旋路径规划器 - 另一种清扫模式"""
    
    def __init__(self, grid_size: float = 1.0, spiral_step: float = 0.5):
        super().__init__(grid_size)
        self.spiral_step = spiral_step
    
    def plan(self, start: Point, goal: Point, bounds: Tuple[Point, Point]) -> List[Point]:
        """生成螺旋清扫路径"""
        min_bound, max_bound = bounds
        
        # 计算螺旋中心
        center = Point(
            (min_bound.x + max_bound.x) / 2,
            (min_bound.y + max_bound.y) / 2
        )
        
        # 生成螺旋路径
        spiral_path = self._generate_spiral_path(center, min_bound, max_bound)
        
        # 从起点连接到螺旋路径
        if spiral_path:
            start_path = self._connect_points(start, spiral_path[0], bounds)
            if start_path:
                spiral_path = start_path[:-1] + spiral_path
            
            # 从螺旋路径到终点
            end_path = self._connect_points(spiral_path[-1], goal, bounds)
            if end_path:
                spiral_path.extend(end_path[1:])
        
        return spiral_path
    
    def _generate_spiral_path(self, center: Point, min_bound: Point, max_bound: Point) -> List[Point]:
        """生成螺旋路径"""
        path = []
        angle = 0
        radius = self.spiral_step
        
        while radius < max(center.distance_to(min_bound), center.distance_to(max_bound)):
            # 计算当前角度对应的点
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            point = Point(x, y)
            
            # 检查是否在边界内且无碰撞
            if (min_bound.x <= x <= max_bound.x and 
                min_bound.y <= y <= max_bound.y and 
                not self.is_collision(point)):
                path.append(point)
            
            # 更新角度和半径
            angle += self.spiral_step / radius
            radius += self.spiral_step / (2 * math.pi)
        
        return path
    
    def _connect_points(self, start: Point, end: Point, bounds: Tuple[Point, Point]) -> List[Point]:
        """连接两个点"""
        if self._is_line_clear(start, end, bounds):
            return [start, end]
        else:
            astar = AStar(self.grid_size)
            astar.obstacles = self.obstacles.copy()
            return astar.plan(start, end, bounds)

class PathPlanningDemo:
    """路径规划演示类"""
    
    def __init__(self, bounds: Tuple[Point, Point]):
        self.bounds = bounds
        self.planners = {
            'A*': AStar(),
            'Dijkstra': Dijkstra(),
            'RRT': RRT(),
            'RRT*': RRTStar(),
            'JPS': JPS(),
            'Zigzag': ZigzagPlanner(),
            'Spiral': SpiralPlanner()
        }
    
    def add_obstacle(self, center: Point, radius: float):
        """为所有规划器添加障碍物"""
        for planner in self.planners.values():
            planner.add_obstacle(center, radius)
    
    def add_rectangular_obstacle(self, top_left: Point, width: float, height: float):
        """为所有规划器添加矩形障碍物"""
        for planner in self.planners.values():
            planner.add_rectangular_obstacle(top_left, width, height)
    
    def plan_path(self, start: Point, goal: Point, algorithm: str = 'A*') -> List[Point]:
        """使用指定算法规划路径"""
        if algorithm not in self.planners:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return self.planners[algorithm].plan(start, goal, self.bounds)
    
    def compare_algorithms(self, start: Point, goal: Point) -> Dict[str, List[Point]]:
        """比较所有算法的路径规划结果"""
        results = {}
        for name, planner in self.planners.items():
            results[name] = planner.plan(start, goal, self.bounds)
        return results
    
    def visualize(self, start: Point, goal: Point, algorithm: str = 'A*', 
                  show_obstacles: bool = True, show_path: bool = True):
        """可视化路径规划结果"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 绘制边界
        min_bound, max_bound = self.bounds
        ax.set_xlim(min_bound.x - 1, max_bound.x + 1)
        ax.set_ylim(min_bound.y - 1, max_bound.y + 1)
        
        # 绘制障碍物
        if show_obstacles:
            for center, radius in self.planners[algorithm].obstacles:
                circle = patches.Circle(center.to_tuple(), radius, 
                                      color='red', alpha=0.7, label='Obstacles')
                ax.add_patch(circle)
        
        # 绘制起点和终点
        ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
        ax.plot(goal.x, goal.y, 'ro', markersize=10, label='Goal')
        
        # 绘制路径
        if show_path:
            path = self.plan_path(start, goal, algorithm)
            if path:
                path_x = [p.x for p in path]
                path_y = [p.y for p in path]
                ax.plot(path_x, path_y, 'b-', linewidth=2, label=f'{algorithm} Path')
                ax.plot(path_x, path_y, 'bo', markersize=4)
            else:
                print(f"No path found with {algorithm}")
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Path Planning with {algorithm}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
    
    def benchmark_algorithms(self, start: Point, goal: Point, num_runs: int = 10) -> Dict[str, Dict[str, float]]:
        """对算法进行性能基准测试"""
        import time
        
        results = {}
        
        for name, planner in self.planners.items():
            times = []
            path_lengths = []
            
            for _ in range(num_runs):
                start_time = time.time()
                path = planner.plan(start, goal, self.bounds)
                end_time = time.time()
                
                times.append(end_time - start_time)
                
                if path:
                    length = sum(path[i].distance_to(path[i+1]) for i in range(len(path)-1))
                    path_lengths.append(length)
                else:
                    path_lengths.append(float('inf'))
            
            results[name] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'avg_length': np.mean(path_lengths),
                'std_length': np.std(path_lengths),
                'success_rate': sum(1 for l in path_lengths if l != float('inf')) / num_runs
            }
        
        return results

# 使用示例
if __name__ == "__main__":
    # 创建演示环境
    bounds = (Point(0, 0), Point(20, 20))
    demo = PathPlanningDemo(bounds)
    
    # 添加一些障碍物
    demo.add_obstacle(Point(5, 5), 2)
    demo.add_obstacle(Point(15, 10), 1.5)
    demo.add_rectangular_obstacle(Point(8, 8), 4, 2)
    
    # 设置起点和终点
    start = Point(1, 1)
    goal = Point(18, 18)
    
    # 比较不同算法
    print("Comparing path planning algorithms...")
    results = demo.compare_algorithms(start, goal)
    
    for algorithm, path in results.items():
        if path:
            length = sum(path[i].distance_to(path[i+1]) for i in range(len(path)-1))
            print(f"{algorithm}: Path length = {length:.2f}, Points = {len(path)}")
        else:
            print(f"{algorithm}: No path found")
    
    # 可视化A*算法结果
    demo.visualize(start, goal, 'A*')
