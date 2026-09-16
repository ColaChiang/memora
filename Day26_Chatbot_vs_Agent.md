# Day 26：Chatbot 與 Agent 架構檢查

這份文件是 Memora Day 26 的 Architecture Checkpoint。今天不新增 Tool、不修改 Memory System，也不提前實作 Day 27 的 Agent Loop；程式版本維持 `Memora v0.22`。

## 目前的系統定位

Day 25 的 Memora 是一個有狀態、有長期記憶，並能在固定 Workflow 中使用一次 Tool 的 **Tool-using Chatbot**。它已有 Agent 的部分元件，但還不能根據 Observation 反覆選擇下一個 Action，因此尚未形成 Agent Loop。

目前 `run_model_with_tools()` 的控制流程如下：

```text
Model（tool_choice="auto"）
├─ 沒有 Function Call → 回傳 Final Answer
└─ 有 Function Call
   → Application 驗證並執行第一個 Tool Call
   → 將 function_call_output 送回 Model
   → Model（tool_choice="none"）
   → 回傳 Final Answer
```

第二次 Model Request 使用 `tool_choice="none"`，所以模型看見 Tool Result 後只能產生答案，不能再選擇另一個 Action。主要路徑仍由 Application 預先決定。

## 四種控制流程

| 類型 | 主要步驟由誰決定 | Tool | 根據結果再次決策 | 停止方式 |
|---|---|---|---|---|
| Chatbot | Application 呼叫模型 | 不一定 | 通常不能 | 模型回傳回答 |
| Tool-using Chatbot | 模型可選 Tool，但流程有限 | 可以 | 視實作而定 | 固定流程結束 |
| Workflow | Python 或 Graph | 可以 | 只走預先定義的分支 | 程式規則 |
| Agent | 模型依 Goal 與 Observation | 通常可以 | 可以反覆決策 | Final Answer、限制或中止條件 |

Memory 與 Agent Control Flow 是不同維度。Memory 決定系統保留與取回哪些狀態；Agent Control Flow 決定下一步 Action，以及何時停止。

## 最小 Agent Run 元件檢查

| 元件 | Memora 對應內容 | Day 26 狀態 |
|---|---|---|
| Goal | Current User Message | 已具備 |
| Instructions | `SYSTEM_PROMPT` | 已具備 |
| State | `input_messages` 與 Output Items | 已具備 |
| Actions | `TOOLS` | 已具備 |
| Executor | `execute_tool_call()` | 已具備 |
| Observation | `function_call_output` | 已具備 |
| Decision Maker | LLM | 已具備 |
| Loop | 重複 Model → Tool → Model | 尚未具備 |
| Stop Condition | Final Answer、Step Limit 或錯誤中止 | 尚未通用化 |

## 保留的安全與狀態邊界

- `TOOL_HANDLERS` 繼續作為可執行 Tool 的 Allowlist。
- Tool Arguments 仍須通過 JSON、型別、必填欄位與長度驗證。
- `count_english_words()` 維持唯讀，Memory 查詢與修改功能不暴露為 Tool。
- Agent Run 繼續採用 Application-managed State，保留完整 `response.output` 與 `function_call_output`。
- 不混用完整 Local Input 與 `previous_response_id`，避免重複 Context。
- 控制流程只依據結構化 Output Items，不依賴自由格式的內部思考文字。

## Day 27 的明確修改範圍

下一步只需集中修改 `run_model_with_tools()`：

1. 在有限步數內重複 Model → Tool → Model。
2. 模型未提出 Function Call 時，將 `output_text` 視為 Final Answer。
3. 每一步保留 Output Items 與 Tool Observation。
4. 累積整個 Agent Run 的 Token Usage。
5. 加入 `MAX_AGENT_STEPS`、正常停止、限制停止與錯誤停止。

Day 26 不實作以上項目，避免把架構辨識與 Runtime Loop 混在同一天。

## 結論

會使用 Tool 不等於已經是 Agent。Day 26 的 Memora 具備 Goal、Action、Executor 與 Observation，但缺少 Observation 後再次 Decision、可重複 Loop、Step Limit 與通用 Stop Condition，因此仍應分類為固定 Workflow 的 Tool-using Chatbot。
