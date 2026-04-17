# 智能家居离线运行环境服务 (Environment)

这个目录负责模拟所有的智能家居设备及其实体逻辑状态。

## 为什么要分离出独立服务？

在 Agent 的进化过程中，为了更清晰地追踪模型发出的每一个“操作指令”，我们将原本直接嵌入 Agent 进程的模拟代码，单独抽取成了一个采用常规 HTTP 通信机制的标准服务端 API (基于 FastAPI)。通过这种设计：
1. **解耦**：大模型 Agent 侧只需维护思考逻辑和 HTTP 调用栈，如同对接一个真实的云服务 IoT 平台。
2. **可视化 Debug**：服务端运行在一个独立进程内，所有被执行的设备状态变更（如开灯、调温度、设备报错）都会以非常直观的 Log `[ENV SERVER]` 形式在终端打印，便于我们了解模型是否真的调用了正确的动作。

## 目录结构
- `server.py`：新引入的 FastAPI 环境驱动服务端程序。
- `devices.py`：灯光、空调、洗衣机等设备状态机。
- `smart_home_env.py`：核心环境逻辑，暴露状态管理。

## 启动指南

1. 首先确保已在项目根目录下安装相关依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 运行环境服务节点：
   ```bash
   python -m environment.server
   ```
   你可以看到控制台显示 `Uvicorn running on http://0.0.0.0:6666`。

3. 在 Agent 端（重新开一个终端执行），当你与助手对话并产生实质上的控制命令时，你的环境平台服务端就能实时捕捉并打印所有被执行的命令。

## 核心接口说明 (Port: 6666)
服务端暴露了这几个便于其他应用调用的 HTTP API：
* `POST /session/{session_id}/reset`: 重新出厂设置。
* `GET /session/{session_id}/state`: 查看全屋设备健康状况。
* `GET /session/{session_id}/events`: 获取消费事件队列（供洗衣机洗完发送弹窗等）。
* `POST /session/{session_id}/action`: 通过 JSON 下发指令 `{'action': {'name': 'tool', 'args': {}}}`。
