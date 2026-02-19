#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python学习平台 - 公网访问启动器
使用 ngrok 创建公网访问链接，分享给朋友访问

使用前需要:
1. 注册 ngrok 账户: https://ngrok.com/signup
2. 获取 authtoken: https://dashboard.ngrok.com/get-started/your-authtoken
3. 运行此脚本，首次使用会提示输入 authtoken
"""

import os
import sys
import subprocess
import time
import json

# 配置文件路径
CONFIG_FILE = os.path.join(os.path.dirname(__file__), '.ngrok_config.json')

def check_dependencies():
    """检查并安装依赖"""
    print("正在检查依赖...")
    
    try:
        import flask
        import flask_cors
    except ImportError:
        print("正在安装 Flask...")
        subprocess.run([sys.executable, "-m", "pip", "install", "flask", "flask-cors"], 
                       capture_output=True)
    
    try:
        from pyngrok import ngrok
    except ImportError:
        print("正在安装 pyngrok...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyngrok"], 
                       capture_output=True)
        from pyngrok import ngrok
    
    return ngrok

def get_authtoken():
    """获取 ngrok authtoken"""
    # 尝试从配置文件读取
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config.get('authtoken')
    return None

def save_authtoken(token):
    """保存 authtoken 到配置文件"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump({'authtoken': token}, f)

def main():
    print("\n" + "=" * 60)
    print("🌐 Python 学习平台 - 公网访问启动器")
    print("=" * 60)
    
    # 检查依赖
    ngrok = check_dependencies()
    
    # 获取 authtoken
    authtoken = get_authtoken()
    if not authtoken:
        print("\n⚠️  首次使用需要配置 ngrok authtoken")
        print("\n📋 获取 authtoken 步骤：")
        print("   1. 访问 https://ngrok.com/signup 注册账户（免费）")
        print("   2. 登录后访问 https://dashboard.ngrok.com/get-started/your-authtoken")
        print("   3. 复制你的 authtoken")
        print()
        authtoken = input("请输入你的 ngrok authtoken: ").strip()
        if authtoken:
            save_authtoken(authtoken)
            print("✅ authtoken 已保存！")
        else:
            print("❌ authtoken 不能为空")
            return
    
    # 配置 ngrok
    ngrok.set_auth_token(authtoken)
    
    print("\n🚀 正在启动服务...")
    
    # 启动 Flask 服务器（后台）
    os.chdir(os.path.dirname(__file__))
    
    # 启动服务器进程
    server_process = subprocess.Popen(
        [sys.executable, "-c", 
         "from app import app; app.run(debug=False, host='0.0.0.0', port=5000)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # 等待服务器启动
    time.sleep(2)
    
    # 创建 ngrok 隧道
    try:
        public_url = ngrok.connect(5000)
        
        print("\n" + "=" * 60)
        print("✅ 服务已启动！")
        print("=" * 60)
        print(f"\n🔗 公网访问地址：")
        print(f"   {public_url}")
        print("\n📤 分享给朋友：")
        print(f"   复制上面的链接发给朋友即可访问")
        print("\n💡 提示：")
        print("   - 链接在服务器运行期间有效")
        print("   - 关闭此窗口将停止服务")
        print("   - 每次启动会生成新的链接")
        print("\n" + "=" * 60)
        print("按 Ctrl+C 停止服务")
        print("=" * 60 + "\n")
        
        # 保持运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 正在停止服务...")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("\n可能的原因：")
        print("  - authtoken 无效")
        print("  - 网络连接问题")
        print("  - 端口 5000 已被占用")
    finally:
        # 清理
        ngrok.disconnect(public_url)
        server_process.terminate()
        print("👋 服务已停止")

if __name__ == "__main__":
    main()
