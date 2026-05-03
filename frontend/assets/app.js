const DEVICE_ORDER = [
  "living_room_light_1",
  "living_room_ac_1",
  "washing_machine_1",
  "living_room_curtain_1",
  "living_room_sensor_1",
  "desk_plug_1",
];

const state = {
  sessionId: localStorage.getItem("qwen4life-session-id") || `web-demo-${Date.now()}`,
  devices: {},
  busy: false,
  debugType: "",
  debugNode: null,
  chatAbortController: null,
};

const els = {
  statusText: document.querySelector("#statusText"),
  sessionId: document.querySelector("#sessionId"),
  resetBtn: document.querySelector("#resetBtn"),
  observedAt: document.querySelector("#observedAt"),
  deviceCount: document.querySelector("#deviceCount"),
  homeStage: document.querySelector("#homeStage"),
  deviceGrid: document.querySelector("#deviceGrid"),
  chatLog: document.querySelector("#chatLog"),
  chatForm: document.querySelector("#chatForm"),
  messageInput: document.querySelector("#messageInput"),
  sendBtn: document.querySelector("#sendBtn"),
  debugToggle: document.querySelector("#debugToggle"),
  eventList: document.querySelector("#eventList"),
  clearEventsBtn: document.querySelector("#clearEventsBtn"),
  sensorTemp: document.querySelector("#sensorTemp"),
  sensorHumidity: document.querySelector("#sensorHumidity"),
  washerStatus: document.querySelector("#washerStatus"),
  plugPower: document.querySelector("#plugPower"),
};

els.sessionId.value = state.sessionId;

els.resetBtn.addEventListener("click", async () => {
  state.sessionId = sanitizeSessionId(els.sessionId.value);
  els.sessionId.value = state.sessionId;
  localStorage.setItem("qwen4life-session-id", state.sessionId);
  await resetSession();
});

els.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = els.messageInput.value.trim();
  if (!message || state.busy) {
    return;
  }
  els.messageInput.value = "";
  await sendChat(message);
});

els.clearEventsBtn.addEventListener("click", () => {
  els.eventList.innerHTML = "";
});

els.debugToggle.addEventListener("change", () => {
  resetDebugStream();
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    els.messageInput.value = button.dataset.prompt || "";
    els.messageInput.focus();
  });
});

async function resetSession() {
  setStatus("正在重置环境");
  abortActiveChat();
  clearConversation();
  const payload = await requestJson(`/api/session/${state.sessionId}/reset`, { method: "POST" });
  renderState(payload.state);
  addEventItems([{ type: "session_reset", source: "web", occurred_at: new Date().toISOString() }]);
  setStatus("环境已就绪");
}

async function refreshState() {
  try {
    const payload = await requestJson(`/api/session/${state.sessionId}/state`);
    renderState(payload.state);
    setStatus(state.busy ? "Agent 正在处理" : "环境同步正常");
  } catch (error) {
    setStatus(`状态同步失败：${error.message}`);
  }
}

async function refreshEvents() {
  try {
    const payload = await requestJson(`/api/session/${state.sessionId}/events`);
    addEventItems(payload.events || []);
  } catch {
    // The first load may happen before the session exists.
  }
}

async function sendChat(message) {
  state.busy = true;
  els.sendBtn.disabled = true;
  state.chatAbortController = new AbortController();
  setStatus("Agent 正在处理");

  addMessage("user", message);
  const assistantMessage = addMessage("assistant", "");
  showThinking(assistantMessage);
  const answerRenderer = createAnswerRenderer(assistantMessage);
  resetDebugStream();

  try {
    await fetchSse(`/api/agent/${state.sessionId}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        verbose: els.debugToggle.checked,
      }),
      signal: state.chatAbortController.signal,
      onEvent(event) {
        handleAgentEvent(event, answerRenderer, assistantMessage);
      },
    });
  } catch (error) {
    if (error.name !== "AbortError") {
      clearThinking(assistantMessage);
      assistantMessage.textContent = `请求失败：${error.message}`;
    }
  } finally {
    state.busy = false;
    els.sendBtn.disabled = false;
    state.chatAbortController = null;
    setStatus("环境同步正常");
    refreshState();
  }
}

function handleAgentEvent(event, answerRenderer, assistantMessage) {
  const payload = event.data || {};
  const content = payload.content || "";

  if (event.type === "content" || event.type === "reasoning") {
    if (els.debugToggle.checked) {
      appendDebugChunk(event.type, content);
    }
    answerRenderer.feed(event.type, content);
    return;
  }

  if (event.type === "action_start") {
    answerRenderer.reset();
    addDebugEvent("action_start", content);
    return;
  }

  if (event.type === "observation") {
    addDebugEvent("observation", content);
    return;
  }

  if (event.type === "final_reply") {
    if (!answerRenderer.hasPrintedAnswer && content.trim()) {
      clearThinking(assistantMessage);
      assistantMessage.textContent = content.trim();
    }
    return;
  }

  if (event.type === "state") {
    renderState(payload.state);
    return;
  }

  if (event.type === "events") {
    addEventItems(payload.events || []);
    return;
  }

  if (event.type === "error") {
    clearThinking(assistantMessage);
    assistantMessage.textContent = content || "Agent 返回错误。";
    addDebugEvent("error", content);
  }
}

function createAnswerRenderer(target) {
  return {
    buffer: "",
    hasPrintedAnswer: false,
    feed(type, text) {
      if (type !== "content") {
        return;
      }
      if (this.hasPrintedAnswer) {
        clearThinking(target);
        target.textContent += text;
        scrollChat();
        return;
      }

      this.buffer += text;
      const answerIndex = this.buffer.indexOf("Answer:");
      if (answerIndex === -1) {
        return;
      }

      this.hasPrintedAnswer = true;
      const answerText = this.buffer.slice(answerIndex + "Answer:".length).trimStart();
      if (answerText) {
        clearThinking(target);
        target.textContent = answerText;
      }
      this.buffer = "";
      scrollChat();
    },
    reset() {
      this.buffer = "";
      this.hasPrintedAnswer = false;
      showThinking(target);
    },
  };
}

function renderState(observation) {
  if (!observation) {
    return;
  }

  state.devices = observation.devices || {};
  els.observedAt.textContent = observation.observed_at || "--";
  els.deviceCount.textContent = `${Object.keys(state.devices).length} 台设备`;
  renderHomeStage(state.devices);
  renderDeviceGrid(state.devices);
}

function renderHomeStage(devices) {
  const light = devices.living_room_light_1 || {};
  const ac = devices.living_room_ac_1 || {};
  const curtain = devices.living_room_curtain_1 || {};
  const sensor = devices.living_room_sensor_1 || {};
  const washer = devices.washing_machine_1 || {};
  const plug = devices.desk_plug_1 || {};

  els.homeStage.classList.toggle("light-on", Boolean(light.is_on));
  els.homeStage.classList.toggle("ac-on", Boolean(ac.is_on));
  els.homeStage.classList.toggle("washer-running", washer.status === "running");
  els.homeStage.classList.toggle("plug-on", Boolean(plug.is_on));
  els.homeStage.style.setProperty("--light-opacity", `${Math.max(0.18, Number(light.brightness || 0) / 100)}`);
  els.homeStage.style.setProperty("--curtain-open", `${Number(curtain.position_percent || 0)}`);

  els.sensorTemp.textContent = `${sensor.temperature ?? "--"}°`;
  els.sensorHumidity.textContent = `${sensor.humidity ?? "--"}%`;
  els.washerStatus.textContent = washer.status ? formatWasherStatus(washer) : "空闲";
  els.plugPower.textContent = `${plug.power_watts ?? 0} W`;
}

function renderDeviceGrid(devices) {
  els.deviceGrid.innerHTML = "";
  const deviceIds = DEVICE_ORDER.filter((id) => devices[id]).concat(
    Object.keys(devices).filter((id) => !DEVICE_ORDER.includes(id)),
  );
  deviceIds.forEach((deviceId) => {
    els.deviceGrid.appendChild(renderDeviceCard(deviceId, devices[deviceId]));
  });
}

function renderDeviceCard(deviceId, device) {
  const card = document.createElement("article");
  card.className = "device-card";
  card.innerHTML = `
    <header>
      <div>
        <h3>${escapeHtml(device.name || deviceId)}</h3>
        <small>${escapeHtml(deviceId)}</small>
      </div>
      <span class="${statusClass(device)}">${statusText(device)}</span>
    </header>
    <div class="device-facts">${deviceFacts(device).map((fact) => `<span>${escapeHtml(fact)}</span>`).join("")}</div>
  `;
  card.appendChild(renderControls(deviceId, device));
  return card;
}

function renderControls(deviceId, device) {
  const row = document.createElement("div");
  row.className = "control-row";

  if (device.device_type === "light") {
    row.append(button("开", () => manualAction("light", deviceId, "turn_on")));
    row.append(button("关", () => manualAction("light", deviceId, "turn_off")));
    const input = numberInput(device.brightness || 0, 0, 100);
    row.append(input, button("亮度", () => manualAction("light", deviceId, "set_brightness", { brightness: Number(input.value) })));
  } else if (device.device_type === "ac") {
    row.append(button("开", () => manualAction("ac", deviceId, "turn_on")));
    row.append(button("关", () => manualAction("ac", deviceId, "turn_off")));
    const input = numberInput(device.target_temperature || 24, 16, 30);
    row.append(input, button("温度", () => manualAction("ac", deviceId, "set_temperature", { temperature: Number(input.value) })));
  } else if (device.device_type === "curtain") {
    row.append(button("开", () => manualAction("curtain", deviceId, "open")));
    row.append(button("关", () => manualAction("curtain", deviceId, "close")));
    const input = numberInput(device.position_percent || 0, 0, 100);
    row.append(input, button("开合", () => manualAction("curtain", deviceId, "set_position", { position_percent: Number(input.value) })));
  } else if (device.device_type === "washing_machine") {
    row.append(button("启动", () => manualAction("washing_machine", deviceId, "start_wash", { program: "quick", duration_seconds: 120 })));
    row.append(button("暂停", () => manualAction("washing_machine", deviceId, "pause")));
    row.append(button("继续", () => manualAction("washing_machine", deviceId, "resume")));
    row.append(button("取消", () => manualAction("washing_machine", deviceId, "cancel")));
  } else if (device.device_type === "smart_plug") {
    row.append(button("开", () => manualAction("smart_plug", deviceId, "turn_on", { power_watts: 12 })));
    row.append(button("关", () => manualAction("smart_plug", deviceId, "turn_off")));
  } else {
    const label = document.createElement("span");
    label.textContent = "只读";
    row.append(label);
  }
  return row;
}

async function manualAction(device, target, command, params = {}) {
  try {
    const payload = await requestJson(`/api/session/${state.sessionId}/action`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device, target, command, params }),
    });
    renderState(payload.state);
    addEventItems(payload.events || []);
    if (payload.result && payload.result.error) {
      setStatus(payload.result.error.message || "手动控制失败");
    }
  } catch (error) {
    setStatus(`手动控制失败：${error.message}`);
  }
}

async function fetchSse(url, options) {
  const { onEvent, ...fetchOptions } = options;
  const response = await fetch(url, fetchOptions);
  if (!response.ok || !response.body) {
    throw new Error(`HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    parts.forEach((part) => {
      const parsed = parseSse(part);
      if (parsed) {
        onEvent(parsed);
      }
    });
  }
}

function parseSse(raw) {
  const lines = raw.split(/\r?\n/);
  const eventLine = lines.find((line) => line.startsWith("event:"));
  const dataLines = lines.filter((line) => line.startsWith("data:"));
  if (!eventLine || dataLines.length === 0) {
    return null;
  }
  const type = eventLine.slice("event:".length).trim();
  const dataText = dataLines.map((line) => line.slice("data:".length).trimStart()).join("\n");
  return {
    type,
    data: JSON.parse(dataText),
  };
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

function addMessage(role, text) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = text;
  els.chatLog.appendChild(message);
  scrollChat();
  return message;
}

function showThinking(node) {
  node.classList.add("thinking");
  node.innerHTML = '<span class="thinking-label">思考中</span><span class="thinking-dots" aria-hidden="true"><i></i><i></i><i></i></span>';
  scrollChat();
}

function clearThinking(node) {
  if (!node.classList.contains("thinking")) {
    return;
  }
  node.classList.remove("thinking");
  node.textContent = "";
}

function appendDebugChunk(type, text) {
  if (!els.debugToggle.checked && type !== "error") {
    return;
  }
  if (state.debugType !== type || !state.debugNode) {
    state.debugType = type;
    state.debugNode = createDebugNode(type);
  }
  state.debugNode.textContent += text;
  scrollChat();
}

function addDebugEvent(type, text) {
  if (!els.debugToggle.checked && type !== "error") {
    return;
  }
  state.debugType = "";
  state.debugNode = createDebugNode(type);
  state.debugNode.textContent += text.trim();
  scrollChat();
}

function createDebugNode(type) {
  const wrapper = document.createElement("div");
  wrapper.className = "debug-log";

  const title = document.createElement("strong");
  title.textContent = type;

  const body = document.createElement("pre");
  wrapper.append(title, body);
  els.chatLog.appendChild(wrapper);
  return body;
}

function resetDebugStream() {
  state.debugType = "";
  state.debugNode = null;
}

function clearConversation() {
  els.chatLog.innerHTML = "";
  els.eventList.innerHTML = "";
  resetDebugStream();
}

function abortActiveChat() {
  if (state.chatAbortController) {
    state.chatAbortController.abort();
  }
  state.busy = false;
  els.sendBtn.disabled = false;
}

function addEventItems(events) {
  events.slice().reverse().forEach((event) => {
    const item = document.createElement("li");
    item.className = "event-item";
    item.innerHTML = `
      <strong>${escapeHtml(event.type || "event")}</strong>
      <span>${escapeHtml(event.source || "system")} · ${escapeHtml(event.occurred_at || "")}</span>
    `;
    els.eventList.prepend(item);
  });
}

function deviceFacts(device) {
  switch (device.device_type) {
    case "light":
      return [`亮度 ${device.brightness ?? 0}`, device.is_on ? "开启" : "关闭"];
    case "ac":
      return [device.is_on ? "开启" : "关闭", modeLabel(device.mode), `${device.target_temperature}°C`, `风速 ${device.fan_speed}`];
    case "washing_machine":
      return [formatWasherStatus(device), `程序 ${device.program}`, `剩余 ${formatDuration(device.remaining_seconds || 0)}`];
    case "curtain":
      return [`开合度 ${device.position_percent ?? 0}%`];
    case "temperature_humidity_sensor":
      return [`温度 ${device.temperature}°C`, `湿度 ${device.humidity}%`];
    case "smart_plug":
      return [device.is_on ? "开启" : "关闭", `功率 ${device.power_watts ?? 0}W`];
    default:
      return [`类型 ${device.device_type || "unknown"}`];
  }
}

function statusText(device) {
  if (!device.online) {
    return "离线";
  }
  if (device.is_on || device.status === "running") {
    return "运行";
  }
  if (device.status && device.status !== "idle") {
    return formatWasherStatus(device);
  }
  return "在线";
}

function statusClass(device) {
  const classes = ["status-pill"];
  if (device.is_on || device.status === "running") {
    classes.push("on");
  }
  if (!device.online || device.status === "paused") {
    classes.push("warn");
  }
  return classes.join(" ");
}

function button(text, onClick) {
  const element = document.createElement("button");
  element.type = "button";
  element.textContent = text;
  element.addEventListener("click", onClick);
  return element;
}

function numberInput(value, min, max) {
  const input = document.createElement("input");
  input.type = "number";
  input.min = String(min);
  input.max = String(max);
  input.value = String(value);
  return input;
}

function formatWasherStatus(device) {
  const map = {
    idle: "空闲",
    running: "运行中",
    paused: "已暂停",
    completed: "已完成",
    cancelled: "已取消",
  };
  return map[device.status] || device.status || "空闲";
}

function formatDuration(seconds) {
  const value = Number(seconds || 0);
  if (value <= 0) {
    return "0 秒";
  }
  const minutes = Math.floor(value / 60);
  const remain = value % 60;
  if (minutes <= 0) {
    return `${remain} 秒`;
  }
  return remain ? `${minutes} 分 ${remain} 秒` : `${minutes} 分钟`;
}

function modeLabel(mode) {
  return {
    cool: "制冷",
    heat: "制热",
    fan: "送风",
    dry: "除湿",
  }[mode] || mode || "未知";
}

function setStatus(text) {
  els.statusText.textContent = text;
}

function sanitizeSessionId(value) {
  return value.trim() || `web-demo-${Date.now()}`;
}

function scrollChat() {
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

resetSession();
setInterval(refreshState, 1600);
setInterval(refreshEvents, 2000);
