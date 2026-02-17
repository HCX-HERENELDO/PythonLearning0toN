# Socket 编程

## 目录
1. [TCP 编程](#1-tcp-编程)
2. [UDP 编程](#2-udp-编程)
3. [简单聊天室](#3-简单聊天室)

---

## 1. TCP 编程

### TCP 客户端

```python
import socket

# 创建 TCP socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 连接服务器
client.connect(('localhost', 8888))

# 发送数据
client.send(b'Hello, Server!')

# 接收数据
data = client.recv(1024)
print(f"收到: {data.decode()}")

# 关闭连接
client.close()
```

### TCP 服务器

```python
import socket

# 创建 TCP socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定地址和端口
server.bind(('localhost', 8888))

# 监听连接
server.listen(5)
print("服务器启动，等待连接...")

while True:
    # 接受客户端连接
    client_socket, address = server.accept()
    print(f"客户端连接: {address}")
    
    # 接收数据
    data = client_socket.recv(1024)
    print(f"收到: {data.decode()}")
    
    # 发送响应
    client_socket.send(b'Hello, Client!')
    
    # 关闭客户端连接
    client_socket.close()
```

---

## 2. UDP 编程

### UDP 客户端

```python
import socket

# 创建 UDP socket
client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 发送数据（不需要连接）
client.sendto(b'Hello, UDP Server!', ('localhost', 8889))

# 接收数据
data, server = client.recvfrom(1024)
print(f"收到: {data.decode()}")

client.close()
```

### UDP 服务器

```python
import socket

# 创建 UDP socket
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 绑定地址和端口
server.bind(('localhost', 8889))
print("UDP 服务器启动...")

while True:
    # 接收数据
    data, client_address = server.recvfrom(1024)
    print(f"收到来自 {client_address}: {data.decode()}")
    
    # 发送响应
    server.sendto(b'Hello, UDP Client!', client_address)
```

---

## 3. 简单聊天室

### 服务器代码

```python
import socket
import threading

clients = []

def handle_client(client_socket, address):
    """处理客户端消息"""
    while True:
        try:
            data = client_socket.recv(1024)
            if not data:
                break
            
            # 广播给所有客户端
            message = f"{address}: {data.decode()}"
            for client in clients:
                if client != client_socket:
                    client.send(message.encode())
        except:
            break
    
    # 移除断开的客户端
    clients.remove(client_socket)
    client_socket.close()
    print(f"客户端断开: {address}")

# 创建服务器
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 8890))
server.listen(5)
print("聊天室服务器启动...")

while True:
    client_socket, address = server.accept()
    clients.append(client_socket)
    print(f"客户端连接: {address}")
    
    # 为每个客户端创建线程
    thread = threading.Thread(target=handle_client, args=(client_socket, address))
    thread.start()
```

### 客户端代码

```python
import socket
import threading

def receive_messages(client_socket):
    """接收消息线程"""
    while True:
        try:
            data = client_socket.recv(1024)
            if not data:
                break
            print(data.decode())
        except:
            break

# 连接服务器
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 8890))

# 启动接收线程
thread = threading.Thread(target=receive_messages, args=(client,))
thread.daemon = True
thread.start()

# 发送消息
print("输入消息（输入 'quit' 退出）:")
while True:
    message = input()
    if message == 'quit':
        break
    client.send(message.encode())

client.close()
```

---

## 练习题

1. 实现一个简单的文件传输程序
2. 创建一个简单的 HTTP 服务器
3. 实现多人群聊功能
