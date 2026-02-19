#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python学习平台 - 启动脚本
运行此脚本将启动服务器并提供公网访问链接
"""

import os
import sys
import subprocess
import webbrowser
import time

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def check_flask():
    """检查 Flask 是否已安装"""
    try:
        import flask
        return True
    except ImportError:
        return False

def install_dependencies():
    """安装必要的依赖"""
    print("正在安装依赖...")
    subprocess.run([sys.executable, "-m", "pip", "install", "flask", "flask-cors"], 
                   capture_output=True)
    print("依赖安装完成！")

def start_server():
    """启动 Flask 服务器"""
    os.chdir(SCRIPT_DIR)
    
    print("\n" + "=" * 60)
    print("🐍 Python 学习交互平台")
    print("=" * 60)
    print("\n正在启动服务器...")
    print("\n📋 访问地址：")
    print("   本地访问: http://localhost:5000")
    print("   局域网访问: http://<你的IP地址>:5000")
    print("\n" + "=" * 60)
    print("提示: 按 Ctrl+C 停止服务器")
    print("=" * 60 + "\n")
    
    # 导入并运行 Flask 应用
    from app import app
    app.run(debug=False, host='0.0.0.0', port=5000)

def main():
    print("\n🚀 Python 学习平台启动器\n")
    
    # 检查依赖
    if not check_flask():
        install_dependencies()
    
    # 启动服务器
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n\n👋 服务器已停止。感谢使用！")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        input("\n按回车键退出...")

if __name__ == "__main__":
    main()