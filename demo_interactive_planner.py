#!/usr/bin/env python3

"""
交互式路径规划程序演示脚本
展示如何使用不同的算法进行路径规划
"""

import os
import sys
import subprocess
import time

def run_demo():
    """运行演示"""
    print("=" * 60)
    print("交互式路径规划程序演示")
    print("=" * 60)
    
    # 查找可用的户型图
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        import glob
        image_files.extend(glob.glob(f"/media/lixiang/matrix/Documents/code/opensource/floor_plane/{ext}"))
    
    if not image_files:
        print("未找到户型图文件！")
        print("请将户型图文件放在项目根目录下")
        return
    
    print(f"找到户型图文件: {image_files}")
    
    # 选择图像文件
    if len(image_files) == 1:
        selected_image = image_files[0]
        print(f"使用图像: {selected_image}")
    else:
        print("\n请选择要使用的户型图:")
        for i, img in enumerate(image_files):
            print(f"{i+1}. {os.path.basename(img)}")
        
        try:
            choice = int(input("请输入选择 (1-{}): ".format(len(image_files)))) - 1
            if 0 <= choice < len(image_files):
                selected_image = image_files[choice]
            else:
                print("无效选择，使用第一个文件")
                selected_image = image_files[0]
        except ValueError:
            print("无效输入，使用第一个文件")
            selected_image = image_files[0]
    
    # 选择算法
    algorithms = ['A*', 'Dijkstra', 'RRT', 'RRT*', 'JPS']
    print(f"\n可用算法: {', '.join(algorithms)}")
    
    try:
        choice = input("请选择算法 (默认A*): ").strip()
        if choice in algorithms:
            selected_algorithm = choice
        else:
            selected_algorithm = 'A*'
    except KeyboardInterrupt:
        print("\n演示取消")
        return
    
    print(f"使用算法: {selected_algorithm}")
    
    # 显示使用说明
    print("\n" + "=" * 60)
    print("使用说明:")
    print("=" * 60)
    print("• 左键点击：设置起点（绿色圆点）")
    print("• 右键点击：设置终点（红色圆点）")
    print("• 按P键：生成路径")
    print("• 按C键：清除路径")
    print("• 按R键：重新加载地图")
    print("• 算法选择：使用下拉菜单切换算法")
    print("\n注意：起点和终点必须在白色可通行区域内！")
    print("=" * 60)
    
    # 启动程序
    print(f"\n启动交互式路径规划程序...")
    print(f"图像: {os.path.basename(selected_image)}")
    print(f"算法: {selected_algorithm}")
    print("\n程序启动中，请稍候...")
    
    try:
        # 使用subprocess启动程序
        cmd = [sys.executable, 'interactive_path_planner.py', selected_image, '-a', selected_algorithm]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"程序运行错误: {e}")
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"未知错误: {e}")

def show_help():
    """显示帮助信息"""
    print("交互式路径规划程序演示")
    print("=" * 40)
    print("用法:")
    print("  python demo_interactive_planner.py")
    print("")
    print("功能:")
    print("  • 自动查找户型图文件")
    print("  • 交互式选择算法")
    print("  • 启动图形界面程序")
    print("")
    print("支持的图像格式: PNG, JPG, JPEG")
    print("支持的算法: A*, Dijkstra, RRT, RRT*, JPS")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        show_help()
    else:
        run_demo()
