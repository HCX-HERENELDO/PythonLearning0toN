#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python学习交互平台 - 后端服务
作者: Hereneldo
"""

import os
import sys
import re
import json
import subprocess
import tempfile
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)

# ============================
# 内容解析器
# ============================

class ContentParser:
    """解析Python和Markdown学习文件"""
    
    def __init__(self, base_dir):
        self.base_dir = base_dir
    
    def get_modules(self):
        """获取所有学习模块"""
        modules = []
        for item in sorted(os.listdir(self.base_dir)):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path) and re.match(r'^\d+_', item):
                # 解析模块编号和名称
                match = re.match(r'^(\d+)_(.+)$', item)
                if match:
                    num, name = match.groups()
                    lessons = self._get_lessons(item_path)
                    modules.append({
                        'id': int(num),
                        'name': name,
                        'path': item,
                        'lesson_count': len(lessons),
                        'lessons': lessons
                    })
        return sorted(modules, key=lambda x: x['id'])
    
    def _get_lessons(self, module_path):
        """获取模块下的所有课程"""
        lessons = []
        for item in sorted(os.listdir(module_path)):
            if item.endswith(('.py', '.md')) and not item.startswith('__'):
                item_path = os.path.join(module_path, item)
                match = re.match(r'^(\d+)_(.+?)(?:\.py|\.md)$', item)
                if match:
                    num, name = match.groups()
                    lessons.append({
                        'id': int(num),
                        'name': name,
                        'filename': item,
                        'type': 'py' if item.endswith('.py') else 'md',
                        'path': item_path
                    })
        return sorted(lessons, key=lambda x: x['id'])
    
    def parse_python_file(self, file_path):
        """解析Python文件内容"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        result = {
            'metadata': {},
            'sections': [],
            'exercises': '',
            'summary': '',
            'raw_code': content
        }
        
        # 提取模块信息
        meta_pattern = r'"""[\s=]+\n模块名称：(.+?)\n学习目标：(.+?)\nPyCharm 技巧：(.+?)\n[\s=]+"""'
        meta_match = re.search(meta_pattern, content, re.DOTALL)
        if meta_match:
            result['metadata'] = {
                'module_name': meta_match.group(1).strip(),
                'learning_goal': meta_match.group(2).strip(),
                'pycharm_tip': meta_match.group(3).strip()
            }
        
        # 提取章节（基于分隔线）- 改进正则
        section_pattern = r'# ={20,}\n# (第.+?部分)：(.+?)\n# ={20,}\n(.*?)(?=# ={20,}\n# 练习题|# ={20,}\n# 本节小结|$)'
        sections = re.findall(section_pattern, content, re.DOTALL)
        
        for idx, (section_num, section_title, section_content) in enumerate(sections):
            section_data = {
                'id': idx + 1,
                'number': section_num,
                'title': section_title,
                'content': section_content.strip(),
                'subsections': [],
                'content_blocks': []
            }
            
            # 提取小节
            sub_pattern = r'# -{20,}\n# (\d+\.\d+) (.+?)\n# -{20,}\n(.*?)(?=# -{20,}|# ={20,}|$)'
            subsections = re.findall(sub_pattern, section_content, re.DOTALL)
            
            for sub_num, sub_title, sub_content in subsections:
                # 解析小节内容块
                content_blocks = self._parse_content_blocks(sub_content)
                section_data['subsections'].append({
                    'number': sub_num,
                    'title': sub_title,
                    'content': sub_content.strip(),
                    'content_blocks': content_blocks
                })
            
            # 如果没有小节，直接解析章节内容
            if not section_data['subsections']:
                section_data['content_blocks'] = self._parse_content_blocks(section_content)
            
            result['sections'].append(section_data)
        
        # 提取练习题
        exercise_pattern = r'# ={20,}\n# 练习题\n# ={20,}\n"""([\s\S]*?)"""'
        exercise_match = re.search(exercise_pattern, content)
        if exercise_match:
            result['exercises'] = exercise_match.group(1).strip()
        
        # 提取小结
        summary_pattern = r'# ={20,}\n# 本节小结\n# ={20,}\n"""([\s\S]*?)"""'
        summary_match = re.search(summary_pattern, content)
        if summary_match:
            result['summary'] = summary_match.group(1).strip()
        
        return result
    
    def _parse_content_blocks(self, content):
        """解析内容块（概念讲解、代码、技巧等）"""
        blocks = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 概念讲解块
            if line.strip() == '"""':
                concept_lines = []
                i += 1
                while i < len(lines) and lines[i].strip() != '"""':
                    concept_lines.append(lines[i])
                    i += 1
                concept_text = '\n'.join(concept_lines).strip()
                if concept_text:
                    # 检查是否是概念讲解
                    if concept_text.startswith('【概念讲解】'):
                        blocks.append({
                            'type': 'concept',
                            'content': concept_text.replace('【概念讲解】', '').strip()
                        })
                    else:
                        blocks.append({
                            'type': 'text',
                            'content': concept_text
                        })
                i += 1
                continue
            
            # PyCharm 技巧块
            if line.strip().startswith('# 【PyCharm 技巧】'):
                tip_lines = [line.strip().replace('# 【PyCharm 技巧】', '').strip()]
                i += 1
                while i < len(lines) and lines[i].strip().startswith('#') and not lines[i].strip().startswith('# =') and not lines[i].strip().startswith('# -') and not lines[i].strip().startswith('# 【'):
                    tip_lines.append(lines[i].strip().lstrip('# ').strip())
                    i += 1
                blocks.append({
                    'type': 'tip',
                    'content': '\n'.join(tip_lines).strip()
                })
                continue
            
            # 注意事项块
            if line.strip().startswith('# 【注意】'):
                note_lines = [line.strip().replace('# 【注意】', '').strip()]
                i += 1
                while i < len(lines) and lines[i].strip().startswith('#') and not lines[i].strip().startswith('# =') and not lines[i].strip().startswith('# -') and not lines[i].strip().startswith('# 【'):
                    note_lines.append(lines[i].strip().lstrip('# ').strip())
                    i += 1
                blocks.append({
                    'type': 'note',
                    'content': '\n'.join(note_lines).strip()
                })
                continue
            
            # 代码块（非注释行，非空行）
            if line.strip() and not line.strip().startswith('#') and not line.strip() == '"""':
                code_lines = [line]
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    # 继续收集代码行
                    if next_line.strip() and not next_line.strip().startswith('# ') and not next_line.strip().startswith('# =') and not next_line.strip().startswith('# -') and not next_line.strip() == '"""':
                        # 检查是否是独立的注释行（不在代码块内）
                        if next_line.strip().startswith('#') and not next_line.strip().startswith('# 【'):
                            # 检查是否是行内注释
                            if not any(c in next_line for c in ['=', 'print', 'def ', 'class ', 'for ', 'if ', 'return ', 'import ']):
                                break
                        code_lines.append(next_line)
                        i += 1
                    else:
                        break
                
                code_text = '\n'.join(code_lines).strip()
                if code_text:
                    blocks.append({
                        'type': 'code',
                        'content': code_text
                    })
                continue
            
            # 普通注释行（作为说明文字）
            if line.strip().startswith('#') and not line.strip().startswith('# =') and not line.strip().startswith('# -') and not line.strip().startswith('# 【'):
                text_lines = [line.strip().lstrip('# ').strip()]
                i += 1
                while i < len(lines) and lines[i].strip().startswith('#') and not lines[i].strip().startswith('# =') and not lines[i].strip().startswith('# -') and not lines[i].strip().startswith('# 【'):
                    text_lines.append(lines[i].strip().lstrip('# ').strip())
                    i += 1
                text = '\n'.join(text_lines).strip()
                if text:
                    blocks.append({
                        'type': 'text',
                        'content': text
                    })
                continue
            
            i += 1
        
        return blocks
    
    def parse_markdown_file(self, file_path):
        """解析Markdown文件内容"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        result = {
            'metadata': {},
            'sections': [],
            'raw_content': content
        }
        
        # 提取标题
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            result['metadata']['title'] = title_match.group(1).strip()
        
        # 提取章节（基于 ## 标题）
        section_pattern = r'##\s+(\d+\.?\s*.+?)\n(.*?)(?=\n##\s|$)'
        sections = re.findall(section_pattern, content, re.DOTALL)
        
        for idx, (section_title, section_content) in enumerate(sections):
            result['sections'].append({
                'id': idx + 1,
                'title': section_title.strip(),
                'content': section_content.strip()
            })
        
        return result
    
    def get_lesson_content(self, module_path, lesson_filename):
        """获取课程内容"""
        file_path = os.path.join(self.base_dir, module_path, lesson_filename)
        
        if not os.path.exists(file_path):
            return None
        
        if lesson_filename.endswith('.py'):
            return self.parse_python_file(file_path)
        else:
            return self.parse_markdown_file(file_path)


# ============================
# Python代码执行器
# ============================

class PythonExecutor:
    """安全的Python代码执行器"""
    
    def __init__(self, timeout=10):
        self.timeout = timeout
        self.forbidden_modules = ['os', 'subprocess', 'sys', 'shutil', 'socket', 'pickle']
    
    def execute(self, code):
        """执行Python代码并返回结果"""
        # 安全检查
        for module in self.forbidden_modules:
            if re.search(rf'\bimport\s+{module}\b|\bfrom\s+{module}\b', code):
                return {
                    'success': False,
                    'output': '',
                    'error': f'安全限制：不允许导入 {module} 模块'
                }
        
        try:
            # 创建临时文件执行
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(code)
                temp_file = f.name
            
            # 执行代码
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=os.path.dirname(temp_file)
            )
            
            # 删除临时文件
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'output': result.stdout,
                    'error': ''
                }
            else:
                return {
                    'success': False,
                    'output': result.stdout,
                    'error': result.stderr
                }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': '',
                'error': f'执行超时（限制{self.timeout}秒）'
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': str(e)
            }


# ============================
# 路由定义
# ============================

parser = ContentParser(BASE_DIR)
executor = PythonExecutor()

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/modules')
def api_modules():
    """获取所有模块"""
    modules = parser.get_modules()
    return jsonify(modules)

@app.route('/api/lesson/<path:lesson_path>')
def api_lesson(lesson_path):
    """获取课程内容"""
    # lesson_path 格式: 01_基础语法/01_变量与数据类型.py
    parts = lesson_path.split('/')
    if len(parts) >= 2:
        module_path = parts[0]
        lesson_filename = '/'.join(parts[1:])
        content = parser.get_lesson_content(module_path, lesson_filename)
        if content:
            return jsonify(content)
    return jsonify({'error': '课程未找到'}), 404

@app.route('/api/run', methods=['POST'])
def api_run():
    """执行Python代码"""
    data = request.get_json()
    code = data.get('code', '')
    
    if not code:
        return jsonify({'success': False, 'error': '代码不能为空'})
    
    result = executor.execute(code)
    return jsonify(result)

@app.route('/api/search')
def api_search():
    """搜索内容"""
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    
    results = []
    modules = parser.get_modules()
    
    for module in modules:
        for lesson in module['lessons']:
            if query in lesson['name'].lower() or query in module['name'].lower():
                results.append({
                    'module': module['name'],
                    'lesson': lesson['name'],
                    'path': f"{module['path']}/{lesson['filename']}",
                    'type': lesson['type']
                })
    
    return jsonify(results)


# ============================
# 作者信息
# ============================

AUTHOR_INFO = {
    'name': 'Hereneldo',
    'github': 'HCX-HERENELDO',
    'github_url': 'https://github.com/HCX-HERENELDO',
    'email': 'Hereneldo@163.com',
    'wechat': 'HerineledoHCX',
    'qq': '2156535625',
    'description': 'Python学习者，热爱编程，持续学习中...'
}

@app.route('/api/author')
def api_author():
    """获取作者信息"""
    return jsonify(AUTHOR_INFO)


# ============================
# 启动服务
# ============================

if __name__ == '__main__':
    # 从环境变量获取端口（云平台部署需要）
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    print("=" * 50)
    print("🐍 Python学习交互平台")
    print("=" * 50)
    print(f"📂 项目目录: {BASE_DIR}")
    print(f"🌐 访问地址: http://localhost:{port}")
    print("=" * 50)
    app.run(debug=debug, host='0.0.0.0', port=port)
