# Memora

Memora 是《從 Stateless LLM 到 Agentic Memory：30 天打造會記憶的 AI Agent》系列所實作的個人英文學習助理。

專案會從一個每次請求都彼此獨立的 Stateless Chatbot 開始，逐步加入對話紀錄、短期記憶、語意搜尋、長期記憶、使用者輪廓、記憶生命週期、工具呼叫與 Agent Loop。

## 為什麼從 Stateless 開始？

LLM 只能根據當次 Request 中收到的資訊產生回答。聊天介面看起來能接續先前內容，通常是因為應用程式再次把歷史訊息放進 Context，而不是模型自行保存了對話。

- **Context**：模型這一次產生回答時看得到的資訊。
- **Memory**：被系統保存，並能在之後依需要找回的資訊。

Memora 的目標不是每次都把所有舊資料交給模型，而是逐步學會判斷「什麼值得保存、現在需要哪一段，以及何時應該更新或忘記」。

## 開發路線

```text
Stateless LLM
    ↓
Conversation History
    ↓
Short-term Memory
    ↓
Embedding / Semantic Search
    ↓
Long-term Memory / User Profile
    ↓
Memory Policy / Update / Decay
    ↓
Tool Calling / Agent Loop
    ↓
Agentic Memory
```

## 目前進度

- Day 01：建立專案目標，釐清 Context 與 Memory 的差異。
- Day 02：整理文字如何經過 Message、Context 與 Token 進入 LLM。

## 系列文章

- [Day 01｜AI 真的記得你嗎？從 Stateless LLM 開始](https://ithelp.ithome.com.tw/articles/10404651)
- [Day 02｜LLM 到底怎麼聊天？從 Token、Message 到 Context](https://ithelp.ithome.com.tw/articles/10404828)
