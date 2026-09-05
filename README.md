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
- Day 03：完成 `Memora v0.1`，可在終端機連續提問，但每次 Request 仍彼此獨立。
- Day 04：加入固定的 System Prompt，讓 `Memora v0.2` 具備一致的角色、目標與回答原則。
- Day 05：顯示每次送出的 Request 編號與 Input，直接觀察 while loop 不等於對話記憶。
- Day 06：完成 `Memora v0.3`，在程式執行期間保存 Conversation History，並可用 `history` 查看內容。
- Day 07：完成 `Memora v0.4`，顯示每輪與整個 Session 的 Token Usage。
- Day 08：完成 `Memora v0.5`，用完整 Turn 為單位建立 Sliding Window，並同時限制 Input Token Budget。
- Day 09：完成 `Memora v0.6`，把離開 Window 的舊訊息壓縮成 Conversation Summary。
- Day 10：完成 `Memora v0.7`，把 History、Summary、Token Budget 與 Context 組裝封裝成 `ShortTermMemory`。
- Day 11：完成 `Memora v0.8`，用 `remember` 與 `memories` 區分聊天紀錄和 Memory Candidate。
- Day 12：完成 `Memora v0.9`，用獨立的 LLM 任務從 User Message 自動抽取 Memory Candidate。
- Day 13：完成 `Memora v0.10`，以 Pydantic Schema 與 Structured Output 取得穩定、可驗證的記憶資料。
- Day 14：完成 `Memora v0.11`，替 Memory Candidate 建立 Embedding，並可查看向量與比較 Cosine Similarity。

## 執行方式

需求：Python 3.10 以上，以及可使用 OpenAI API 的金鑰。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell：.venv\Scripts\Activate.ps1
pip install -r requirements.txt
export OPENAI_API_KEY="你的 API Key"
python chatbot.py
```

輸入 `history` 可以查看目前的 Conversation History，輸入 `exit` 即可結束程式。API Key 由環境變數讀取，不會寫進原始碼或 Git repository。

目前的 History 只存在記憶體中；結束程式後便會消失。這是 Day 06 刻意保留的限制，後續才會加入 Context Management 與可持久化的長期記憶。

## 系列文章

- [Day 01｜AI 真的記得你嗎？從 Stateless LLM 開始](https://ithelp.ithome.com.tw/articles/10404651)
- [Day 02｜LLM 到底怎麼聊天？從 Token、Message 到 Context](https://ithelp.ithome.com.tw/articles/10404828)
- [Day 03｜用 Python + LLM API 做第一個 Chatbot](https://ithelp.ithome.com.tw/articles/10405086)
- [Day 04｜Prompt 如何改變 AI？System Prompt 到 Prompt Engineering](https://ithelp.ithome.com.tw/articles/10405261)
- [Day 05｜為什麼 LLM API 聊完就忘？理解 Stateless API](https://ithelp.ithome.com.tw/articles/10405458)
- [Day 06｜Conversation History 是什麼？讓 AI 接得上前一句](https://ithelp.ithome.com.tw/articles/10405749)
- [Day 07｜Context Window 為什麼會爆？Token Limit 的真相](https://ithelp.ithome.com.tw/articles/10405924)
- [Day 08｜Token 太多怎麼辦？Sliding Window 與 Context Management](https://ithelp.ithome.com.tw/articles/10406021)
- [Day 09｜AI 也要做筆記：用 Summarization 壓縮對話](https://ithelp.ithome.com.tw/articles/10406389)
- [Day 10｜Short-term Memory 完成：打造 Stateful Chatbot](https://ithelp.ithome.com.tw/articles/10406390)
- [Day 11｜什麼事情值得 AI 記住？從聊天紀錄到 Memory](https://ithelp.ithome.com.tw/articles/10406392)
- [Day 12｜讓 LLM 自動抽取 Memory：從對話找到重要資訊](https://ithelp.ithome.com.tw/articles/10407127)
- [Day 13｜Structured Output：把 Memory 變成真正可以存的資料](https://ithelp.ithome.com.tw/articles/10407432)
- [Day 14｜Embedding：AI 怎麼知道兩段記憶「很像」？](https://ithelp.ithome.com.tw/articles/10407783)
