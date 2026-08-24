# Day 02：從文字到 LLM Response

使用者在聊天介面輸入一句話後，應用程式不會直接把「畫面上的一句話」原封不動交給模型。一次回應大致會經過以下流程：

```text
使用者輸入
    ↓
建立 User Message
    ↓
組合 System Message 與先前訊息
    ↓
形成 Context
    ↓
Tokenizer 轉換為 Tokens
    ↓
LLM 逐步產生 Tokens
    ↓
組成 Assistant Message
```

## Message 的角色

- `system`／`instructions`：定義助理的角色與回答原則。
- `user`：使用者的需求或問題。
- `assistant`：模型先前產生的回答。

多輪對話之所以能接續，不只是因為程式反覆呼叫 API，而是應用程式把先前的 User Message 與 Assistant Message 再次加入下一次 Context。

## Memora 之後需要處理的內容

目前還沒有可執行程式；這一天先確定後續實作的資料流。未來一次 Request 的 Context 可能包含：

```text
System Instructions
+ Conversation History
+ Relevant Memories
+ Current User Message
```

Conversation History、Memory Store 與 User Profile 由應用程式管理；LLM 只負責根據當次收到的 Context 產生 Response。
