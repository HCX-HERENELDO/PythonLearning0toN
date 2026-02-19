/**
 * Python学习交互平台 - 前端逻辑
 * 作者: Hereneldo
 */

// ============================
// 全局状态
// ============================

const state = {
    modules: [],
    currentModule: null,
    currentLesson: null,
    searchTimeout: null
};

// ============================
// 初始化
// ============================

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initSearch();
    initEditor();
    loadModules();
});

// ============================
// 导航功能
// ============================

function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const page = link.dataset.page;
            showPage(page);
        });
    });
}

function showPage(pageName) {
    // 更新导航状态
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.toggle('active', link.dataset.page === pageName);
    });
    
    // 切换页面
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    const targetPage = document.getElementById(pageName + 'Page');
    if (targetPage) {
        targetPage.classList.add('active');
    }
}

// ============================
// 搜索功能
// ============================

function initSearch() {
    const searchInput = document.getElementById('searchInput');
    const searchModal = document.getElementById('searchModal');
    
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        
        if (state.searchTimeout) {
            clearTimeout(state.searchTimeout);
        }
        
        if (query.length < 2) {
            searchModal.classList.remove('show');
            return;
        }
        
        state.searchTimeout = setTimeout(() => {
            performSearch(query);
        }, 300);
    });
    
    searchInput.addEventListener('focus', () => {
        if (searchInput.value.trim().length >= 2) {
            searchModal.classList.add('show');
        }
    });
    
    // 点击外部关闭搜索结果
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.nav-search') && !e.target.closest('.search-modal')) {
            searchModal.classList.remove('show');
        }
    });
}

async function performSearch(query) {
    const searchModal = document.getElementById('searchModal');
    const searchResults = document.getElementById('searchResults');
    
    try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        const results = await response.json();
        
        if (results.length === 0) {
            searchResults.innerHTML = '<div class="no-results">未找到相关内容</div>';
        } else {
            searchResults.innerHTML = results.map(item => `
                <div class="search-item" onclick="openLesson('${item.path}')">
                    <i class="fas ${item.type === 'py' ? 'fa-file-code' : 'fa-file-alt'}"></i>
                    <div class="search-item-info">
                        <div class="search-item-title">${item.lesson}</div>
                        <div class="search-item-path">${item.module}</div>
                    </div>
                </div>
            `).join('');
        }
        
        searchModal.classList.add('show');
    } catch (error) {
        console.error('搜索失败:', error);
    }
}

function openLesson(path) {
    document.getElementById('searchModal').classList.remove('show');
    document.getElementById('searchInput').value = '';
    showPage('modules');
    
    // 解析路径并打开对应课程
    const parts = path.split('/');
    if (parts.length >= 2) {
        const modulePath = parts[0];
        const lessonFilename = parts.slice(1).join('/');
        loadLessonContent(modulePath, lessonFilename);
    }
}

// ============================
// 模块加载
// ============================

async function loadModules() {
    try {
        const response = await fetch('/api/modules');
        state.modules = await response.json();
        
        renderModuleList();
        renderModulePreview();
        updateStats();
    } catch (error) {
        console.error('加载模块失败:', error);
    }
}

function renderModuleList() {
    const moduleList = document.getElementById('moduleList');
    
    moduleList.innerHTML = state.modules.map(module => `
        <div class="module-item">
            <div class="module-header" onclick="toggleModule(${module.id})">
                <span class="module-num">${module.id}</span>
                <span class="module-name">${module.name}</span>
                <span class="lesson-count">${module.lesson_count}</span>
                <i class="fas fa-chevron-right"></i>
            </div>
            <div class="lesson-list" id="lessons-${module.id}">
                ${module.lessons.map(lesson => `
                    <div class="lesson-item ${lesson.type}" 
                         onclick="loadLessonContent('${module.path}', '${lesson.filename}')">
                        <i class="fas ${lesson.type === 'py' ? 'fa-file-code' : 'fa-file-alt'}"></i>
                        <span>${lesson.name}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `).join('');
}

function renderModulePreview() {
    const modulePreview = document.getElementById('modulePreview');
    
    modulePreview.innerHTML = state.modules.slice(0, 10).map(module => `
        <div class="module-card" onclick="showPage('modules'); toggleModule(${module.id})">
            <span class="module-number">${module.id}</span>
            <h4>${module.name}</h4>
            <p>${module.lesson_count} 节课程</p>
        </div>
    `).join('');
}

function updateStats() {
    const totalLessons = state.modules.reduce((sum, m) => sum + m.lesson_count, 0);
    document.getElementById('moduleCount').textContent = state.modules.length;
    document.getElementById('lessonCount').textContent = totalLessons + '+';
}

// ============================
// 课程内容加载
// ============================

function toggleModule(moduleId) {
    const header = document.querySelector(`.module-item:nth-child(${moduleId}) .module-header, 
                                          .module-item:nth-child(${state.modules.findIndex(m => m.id === moduleId) + 1}) .module-header`);
    const lessonList = document.getElementById(`lessons-${moduleId}`);
    
    if (header && lessonList) {
        header.classList.toggle('expanded');
        lessonList.classList.toggle('show');
    }
}

async function loadLessonContent(modulePath, lessonFilename) {
    const contentArea = document.getElementById('lessonContent');
    
    // 更新选中状态
    document.querySelectorAll('.lesson-item').forEach(item => {
        item.classList.remove('active');
    });
    event.target.closest('.lesson-item')?.classList.add('active');
    
    // 显示加载状态
    contentArea.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    
    try {
        const response = await fetch(`/api/lesson/${modulePath}/${lessonFilename}`);
        const data = await response.json();
        
        if (data.error) {
            contentArea.innerHTML = `<div class="welcome-message"><h3>${data.error}</h3></div>`;
            return;
        }
        
        if (lessonFilename.endsWith('.py')) {
            renderPythonContent(data, contentArea);
        } else {
            renderMarkdownContent(data, contentArea);
        }
        
        // 高亮代码
        document.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
        });
        
    } catch (error) {
        console.error('加载课程内容失败:', error);
        contentArea.innerHTML = '<div class="welcome-message"><h3>加载失败，请重试</h3></div>';
    }
}

function renderPythonContent(data, container) {
    const metadata = data.metadata || {};
    const sections = data.sections || [];
    const exercises = data.exercises;
    const summary = data.summary;
    
    let html = `
        <div class="lesson-header">
            <h1>${metadata.module_name || 'Python 课程'}</h1>
            <div class="lesson-meta">
                ${metadata.learning_goal ? `<span><i class="fas fa-bullseye"></i> 学习目标：${metadata.learning_goal}</span>` : ''}
            </div>
        </div>
    `;
    
    // 渲染元数据
    if (metadata.pycharm_tip) {
        html += `
            <div class="tip-block">
                <h4><i class="fas fa-lightbulb"></i> PyCharm 技巧</h4>
                <p>${metadata.pycharm_tip}</p>
            </div>
        `;
    }
    
    // 渲染章节
    sections.forEach(section => {
        html += `
            <div class="lesson-section">
                <h2><i class="fas fa-bookmark"></i> ${section.number}：${section.title}</h2>
            </div>
        `;
        
        // 渲染小节
        if (section.subsections && section.subsections.length > 0) {
            section.subsections.forEach(sub => {
                html += `
                    <div class="subsection">
                        <h3 class="subsection-title">${sub.number} ${sub.title}</h3>
                        <div class="subsection-content">
                            ${renderContentBlocks(sub.content_blocks || sub.content)}
                        </div>
                    </div>
                `;
            });
        } else if (section.content_blocks && section.content_blocks.length > 0) {
            // 如果没有小节，直接渲染章节内容块
            html += `<div class="subsection-content">${renderContentBlocks(section.content_blocks)}</div>`;
        } else {
            // 兜底：使用旧方法渲染
            html += renderSectionContent(section.content);
        }
    });
    
    // 如果没有解析到章节，显示原始代码
    if (sections.length === 0 && data.raw_code) {
        html += `
            <div class="lesson-section">
                <h2><i class="fas fa-code"></i> 完整代码</h2>
                <div class="code-block">
                    <pre><code class="language-python">${escapeHtml(data.raw_code)}</code></pre>
                </div>
                <button class="btn btn-secondary" onclick="loadCodeToEditor(\`${encodeURIComponent(data.raw_code)}\`)">
                    <i class="fas fa-edit"></i> 在编辑器中打开
                </button>
            </div>
        `;
    }
    
    // 渲染练习题
    if (exercises) {
        html += `
            <div class="exercise-block">
                <h3><i class="fas fa-pencil-alt"></i> 练习题</h3>
                <div class="exercise-content">${formatExerciseText(exercises)}</div>
            </div>
        `;
    }
    
    // 渲染小结
    if (summary) {
        html += `
            <div class="summary-block">
                <h3><i class="fas fa-check-circle"></i> 本节小结</h3>
                <div class="summary-content">${formatSummaryText(summary)}</div>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

function renderContentBlocks(blocks) {
    if (!blocks || blocks.length === 0) return '';
    
    // 如果是字符串（旧格式），使用旧方法
    if (typeof blocks === 'string') {
        return renderSectionContent(blocks);
    }
    
    let html = '';
    
    blocks.forEach(block => {
        switch (block.type) {
            case 'concept':
                html += `
                    <div class="concept-block">
                        <h4><i class="fas fa-book"></i> 概念讲解</h4>
                        <div class="concept-content">${formatText(block.content)}</div>
                    </div>
                `;
                break;
            
            case 'tip':
                html += `
                    <div class="tip-block">
                        <h4><i class="fas fa-lightbulb"></i> PyCharm 技巧</h4>
                        <p>${block.content}</p>
                    </div>
                `;
                break;
            
            case 'note':
                html += `
                    <div class="note-block">
                        <h4><i class="fas fa-exclamation-triangle"></i> 注意事项</h4>
                        <p>${block.content}</p>
                    </div>
                `;
                break;
            
            case 'code':
                html += `
                    <div class="code-block-wrapper">
                        <div class="code-block">
                            <div class="code-actions">
                                <button class="code-action-btn" onclick="copyCode(this)" title="复制代码">
                                    <i class="fas fa-copy"></i>
                                </button>
                                <button class="code-action-btn" onclick="runThisCode(this)" title="运行代码">
                                    <i class="fas fa-play"></i>
                                </button>
                            </div>
                            <pre><code class="language-python">${escapeHtml(block.content)}</code></pre>
                        </div>
                    </div>
                `;
                break;
            
            case 'text':
                html += `<p class="content-text">${formatText(block.content)}</p>`;
                break;
        }
    });
    
    return html;
}

function formatText(text) {
    // 将换行符转换为 <br>，保留格式
    return text
        .split('\n')
        .map(line => {
            // 去除 markdown 语法残留
            line = line.trim();
            
            // 处理标题标记 (### ## #)
            line = line.replace(/^#{1,6}\s*/, '');
            
            // 处理粗体和斜体
            line = line.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
            line = line.replace(/\*(.+?)\*/g, '<em>$1</em>');
            line = line.replace(/__(.+?)__/g, '<strong>$1</strong>');
            line = line.replace(/_(.+?)_/g, '<em>$1</em>');
            
            // 处理行内代码
            line = line.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
            
            // 处理列表项
            if (line.match(/^\d+\.\s/)) {
                return `<span class="list-item">${line.replace(/^\d+\.\s/, '')}</span>`;
            }
            if (line.startsWith('- ')) {
                return `<span class="list-item">${line.substring(2)}</span>`;
            }
            if (line.startsWith('* ')) {
                return `<span class="list-item">${line.substring(2)}</span>`;
            }
            
            return line;
        })
        .join('<br>');
}

function formatExerciseText(text) {
    // 格式化练习题文本
    // 去除可能的 markdown 残留
    text = text.replace(/```python\n?/g, '').replace(/```\n?/g, '');
    
    const exercises = text.split(/【练习\d+】/).filter(s => s.trim());
    let html = '';
    
    exercises.forEach((exercise, index) => {
        if (exercise.trim()) {
            // 清理格式
            let content = exercise.trim()
                .replace(/^[：:]\s*/, '')
                .replace(/\n-/g, '\n•')
                .replace(/\n\d+\./g, '\n▸');
            
            html += `
                <div class="exercise-item">
                    <h4><i class="fas fa-code"></i> 练习 ${index + 1}</h4>
                    <p>${formatText(content)}</p>
                </div>
            `;
        }
    });
    
    // 如果没有匹配到格式，直接显示原文
    if (!html) {
        html = `<div class="exercise-pre">${formatText(text)}</div>`;
    }
    
    return html;
}

function formatSummaryText(text) {
    // 格式化小结文本
    // 去除 markdown 残留
    text = text.replace(/```python\n?/g, '').replace(/```\n?/g, '');
    
    let html = '<div class="summary-items">';
    
    // 按段落分割
    const parts = text.split('\n\n');
    
    parts.forEach(part => {
        part = part.trim();
        if (part.includes('✅') || part.includes('掌握的知识点')) {
            html += `<div class="summary-section knowledge">
                <h4><i class="fas fa-graduation-cap"></i> 掌握的知识点</h4>
                <ul>${formatListItems(part)}</ul>
            </div>`;
        } else if (part.includes('🔧') || part.includes('PyCharm 技巧')) {
            html += `<div class="summary-section tips">
                <h4><i class="fas fa-tools"></i> PyCharm 技巧</h4>
                <ul>${formatListItems(part)}</ul>
            </div>`;
        } else if (part.includes('➡️') || part.includes('下一节')) {
            const nextSection = part.replace(/[➡️下一节：:]/g, '').replace(/➡️/g, '').trim();
            html += `<div class="summary-section next">
                <h4><i class="fas fa-arrow-right"></i> 下一节预告</h4>
                <p>${nextSection}</p>
            </div>`;
        } else if (part.trim()) {
            html += `<p>${formatText(part)}</p>`;
        }
    });
    
    html += '</div>';
    return html;
}

function formatListItems(text) {
    const lines = text.split('\n').filter(line => line.trim().match(/^\d+\./));
    return lines.map(line => {
        let content = line.replace(/^\d+\.\s*/, '').trim();
        // 去除 markdown 语法
        content = content.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        content = content.replace(/\*(.+?)\*/g, '<em>$1</em>');
        return `<li>${content}</li>`;
    }).join('');
}

// 复制代码功能
function copyCode(button) {
    const codeBlock = button.closest('.code-block').querySelector('code');
    const text = codeBlock.textContent;
    
    navigator.clipboard.writeText(text).then(() => {
        button.innerHTML = '<i class="fas fa-check"></i>';
        setTimeout(() => {
            button.innerHTML = '<i class="fas fa-copy"></i>';
        }, 2000);
    });
}

// 运行当前代码
function runThisCode(button) {
    const codeBlock = button.closest('.code-block').querySelector('code');
    const code = codeBlock.textContent;
    
    // 切换到代码练习页面并加载代码
    showPage('playground');
    document.getElementById('codeEditor').value = code;
    document.getElementById('codeEditor').dispatchEvent(new Event('input'));
    
    // 自动运行
    setTimeout(() => runCode(), 500);
}

function renderSectionContent(content) {
    // 提取代码块
    let html = '';
    
    // 处理概念讲解块
    const conceptPattern = /"""[\s]*【概念讲解】([\s\S]*?)"""/g;
    content = content.replace(conceptPattern, (match, concept) => {
        return `<div class="concept-block"><h4>📖 概念讲解</h4><p>${concept.trim()}</p></div>`;
    });
    
    // 处理 PyCharm 技巧
    const tipPattern = /# 【PyCharm 技巧】\s*([\s\S]*?)(?=\n#|\n\n|\n"""|$)/g;
    content = content.replace(tipPattern, (match, tip) => {
        return `<div class="tip-block"><h4>💡 PyCharm 技巧</h4><p>${tip.trim()}</p></div>`;
    });
    
    // 提取并渲染代码
    const codePattern = /```python\n([\s\S]*?)```/g;
    const lines = content.split('\n');
    let inCodeBlock = false;
    let codeBuffer = [];
    let textBuffer = [];
    
    for (const line of lines) {
        // 检查是否是实际代码行（以非#开头且不是空行的Python代码）
        const isCodeLine = line.trim() && 
                          !line.trim().startsWith('#') && 
                          !line.trim().startsWith('"""') &&
                          !line.trim().startsWith("'''") &&
                          (line.includes('=') || 
                           line.includes('print') || 
                           line.includes('def ') ||
                           line.includes('class ') ||
                           line.includes('for ') ||
                           line.includes('if ') ||
                           line.includes('return ') ||
                           line.includes('import ') ||
                           line.match(/^\s*\w+\.\w+/) ||
                           line.match(/^\s*[\[\{]/));
        
        if (isCodeLine) {
            if (textBuffer.length > 0) {
                html += `<p>${textBuffer.join('<br>').trim()}</p>`;
                textBuffer = [];
            }
            codeBuffer.push(line);
        } else {
            if (codeBuffer.length > 0) {
                html += `<div class="code-block"><pre><code class="language-python">${escapeHtml(codeBuffer.join('\n'))}</code></pre></div>`;
                codeBuffer = [];
            }
            if (line.trim() && !line.trim().startsWith('# =====') && !line.trim().startsWith('# -----')) {
                textBuffer.push(line);
            }
        }
    }
    
    // 处理剩余内容
    if (codeBuffer.length > 0) {
        html += `<div class="code-block"><pre><code class="language-python">${escapeHtml(codeBuffer.join('\n'))}</code></pre></div>`;
    }
    if (textBuffer.length > 0) {
        html += `<p>${textBuffer.join('<br>').trim()}</p>`;
    }
    
    return html || `<div class="code-block"><pre><code class="language-python">${escapeHtml(content)}</code></pre></div>`;
}

function renderMarkdownContent(data, container) {
    const metadata = data.metadata || {};
    const sections = data.sections || [];
    
    let html = `
        <div class="lesson-header">
            <h1>${metadata.title || '教程'}</h1>
        </div>
    `;
    
    // 简单的 Markdown 渲染
    if (data.raw_content) {
        let content = data.raw_content;
        
        // 移除主标题
        content = content.replace(/^#\s+.+$/m, '');
        
        // 转换标题
        content = content.replace(/^##\s+(.+)$/gm, '<h2>$1</h2>');
        content = content.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
        content = content.replace(/^####\s+(.+)$/gm, '<h4>$1</h4>');
        
        // 转换代码块
        content = content.replace(/```python\n([\s\S]*?)```/g, '<div class="code-block"><pre><code class="language-python">$1</code></pre></div>');
        content = content.replace(/```\n([\s\S]*?)```/g, '<div class="code-block"><pre><code>$1</code></pre></div>');
        content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // 转换表格
        content = content.replace(/\|(.+)\|/g, (match) => {
            const cells = match.split('|').filter(c => c.trim());
            if (cells.every(c => c.trim().match(/^-+$/))) {
                return ''; // 跳过分隔行
            }
            return '<tr>' + cells.map(c => `<td>${c.trim()}</td>`).join('') + '</tr>';
        });
        content = content.replace(/(<tr>.*<\/tr>)+/g, '<table class="md-table">$&</table>');
        
        // 转换列表
        content = content.replace(/^- (.+)$/gm, '<li>$1</li>');
        content = content.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
        
        // 转换引用
        content = content.replace(/^>\s*(.+)$/gm, '<blockquote>$1</blockquote>');
        
        // 转换分隔线
        content = content.replace(/^---$/gm, '<hr>');
        
        // 转换粗体和斜体
        content = content.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        content = content.replace(/\*(.+?)\*/g, '<em>$1</em>');
        
        // 转换链接
        content = content.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
        
        html += `<div class="markdown-content">${content}</div>`;
    }
    
    container.innerHTML = html;
}

// ============================
// 代码编辑器
// ============================

function initEditor() {
    const editor = document.getElementById('codeEditor');
    const lineNumbers = document.getElementById('lineNumbers');
    
    // 更新行号
    function updateLineNumbers() {
        const lines = editor.value.split('\n').length;
        lineNumbers.innerHTML = Array.from({length: lines}, (_, i) => `<span>${i + 1}</span>`).join('');
    }
    
    editor.addEventListener('input', updateLineNumbers);
    editor.addEventListener('scroll', () => {
        lineNumbers.scrollTop = editor.scrollTop;
    });
    
    // Tab 键支持
    editor.addEventListener('keydown', (e) => {
        if (e.key === 'Tab') {
            e.preventDefault();
            const start = editor.selectionStart;
            const end = editor.selectionEnd;
            editor.value = editor.value.substring(0, start) + '    ' + editor.value.substring(end);
            editor.selectionStart = editor.selectionEnd = start + 4;
            updateLineNumbers();
        }
    });
    
    // 初始化行号
    updateLineNumbers();
}

async function runCode() {
    const code = document.getElementById('codeEditor').value;
    const outputArea = document.getElementById('outputArea');
    
    if (!code.trim()) {
        outputArea.innerHTML = '<span class="output-error">请输入代码后再运行</span>';
        return;
    }
    
    outputArea.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    
    try {
        const response = await fetch('/api/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ code })
        });
        
        const result = await response.json();
        
        if (result.success) {
            outputArea.innerHTML = `<span class="output-success">${escapeHtml(result.output) || '程序执行成功，无输出'}</span>`;
        } else {
            outputArea.innerHTML = `<span class="output-error">${escapeHtml(result.error)}</span>`;
            if (result.output) {
                outputArea.innerHTML += `\n<span class="output-success">${escapeHtml(result.output)}</span>`;
            }
        }
    } catch (error) {
        outputArea.innerHTML = `<span class="output-error">运行失败: ${error.message}</span>`;
    }
}

function clearCode() {
    document.getElementById('codeEditor').value = '';
    document.getElementById('lineNumbers').innerHTML = '<span>1</span>';
}

function clearOutput() {
    document.getElementById('outputArea').innerHTML = `
        <div class="output-placeholder">
            <i class="fas fa-arrow-left"></i>
            <span>运行代码后在此查看结果</span>
        </div>
    `;
}

function loadCodeToEditor(encodedCode) {
    const code = decodeURIComponent(encodedCode);
    showPage('playground');
    document.getElementById('codeEditor').value = code;
    document.getElementById('codeEditor').dispatchEvent(new Event('input'));
}

// ============================
// 代码模板
// ============================

const templates = {
    hello: `# Hello World 示例
print("Hello, World!")
print("欢迎来到Python世界！")

# 使用变量
name = "学习者"
print(f"你好, {name}!")`,
    
    loop: `# 循环示例

# for 循环
print("for 循环示例:")
for i in range(5):
    print(f"  第 {i+1} 次循环")

# while 循环
print("\\nwhile 循环示例:")
count = 0
while count < 3:
    print(f"  count = {count}")
    count += 1

# 列表遍历
print("\\n列表遍历:")
fruits = ["苹果", "香蕉", "橙子"]
for fruit in fruits:
    print(f"  我喜欢吃{fruit}")`,
    
    function: `# 函数定义示例

# 基本函数
def greet(name):
    """问候函数"""
    return f"你好, {name}!"

print(greet("Python学习者"))

# 带默认参数的函数
def power(base, exponent=2):
    """计算幂"""
    return base ** exponent

print(f"2的3次方 = {power(2, 3)}")
print(f"3的平方 = {power(3)}")

# 可变参数
def sum_all(*numbers):
    """计算所有参数的和"""
    return sum(numbers)

print(f"1+2+3+4+5 = {sum_all(1, 2, 3, 4, 5)}")`,
    
    class: `# 类定义示例

class Student:
    """学生类"""
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.grades = []
    
    def add_grade(self, grade):
        """添加成绩"""
        self.grades.append(grade)
    
    def get_average(self):
        """计算平均成绩"""
        if not self.grades:
            return 0
        return sum(self.grades) / len(self.grades)
    
    def __str__(self):
        return f"学生: {self.name}, 年龄: {self.age}, 平均分: {self.get_average():.1f}"

# 创建学生对象
student = Student("张三", 20)
student.add_grade(85)
student.add_grade(90)
student.add_grade(78)

print(student)`,
    
    list: `# 列表操作示例

# 创建列表
numbers = [1, 2, 3, 4, 5]
print(f"原始列表: {numbers}")

# 添加元素
numbers.append(6)
print(f"append(6)后: {numbers}")

# 插入元素
numbers.insert(0, 0)
print(f"insert(0, 0)后: {numbers}")

# 删除元素
numbers.remove(3)
print(f"remove(3)后: {numbers}")

# 列表切片
print(f"前三个元素: {numbers[:3]}")
print(f"最后两个元素: {numbers[-2:]}")

# 列表推导式
squares = [x**2 for x in range(1, 6)]
print(f"1-5的平方: {squares}")

# 过滤
even = [x for x in numbers if x % 2 == 0]
print(f"偶数: {even}")`
};

function loadTemplate(name) {
    const template = templates[name];
    if (template) {
        document.getElementById('codeEditor').value = template;
        document.getElementById('codeEditor').dispatchEvent(new Event('input'));
    }
}

// ============================
// 工具函数
// ============================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 键盘快捷键
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter 运行代码
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (document.getElementById('playgroundPage').classList.contains('active')) {
            e.preventDefault();
            runCode();
        }
    }
    
    // Escape 关闭搜索
    if (e.key === 'Escape') {
        document.getElementById('searchModal').classList.remove('show');
    }
});
