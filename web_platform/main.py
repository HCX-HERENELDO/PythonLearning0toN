#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python学习交互平台 - 入口文件
用于 Railway 部署
"""

import os
import sys

# 设置基础目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 在 Railway 上，内容文件在父目录
BASE_DIR = os.path.dirname(CURRENT_DIR)

# 检查内容是否存在
content_exists = os.path.exists(os.path.join(BASE_DIR, '01_基础语法'))
if not content_exists:
    BASE_DIR = CURRENT_DIR

# 导入并运行 app
from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
