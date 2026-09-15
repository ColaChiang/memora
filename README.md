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
- Day 15：完成 `Memora v0.12`，不用 Vector Database，先以 Cosine Similarity 實作 Top-K Semantic Search。
- Day 16：完成 `Memora v0.13`，改用 Chroma 儲存 Embedding、Metadata，並以 Cosine Distance 執行搜尋。
- Day 17：完成 `Memora v0.14`，用 Chroma PersistentClient 建立可跨程式保存的 `LongTermMemoryStore`。
- Day 18：完成 `Memora v0.15`，在回答前自動檢索、篩選相關記憶，並把它納入 Context Token Budget。
- Day 19：完成 `Memora v0.16`，用持久化 JSON Profile 區分目前使用者設定與可搜尋的過去記憶。
- Day 20：完成 `Memora v0.17`，將 Semantic 與 Episodic Memory 類型一路保留到 Chroma Metadata 與 Retrieval Context。
- Day 21：完成 `Memora v0.18`，在寫入前套用 Memory Policy，分流自動抽取與明確記憶要求，並阻擋敏感或短暫內容。
- Day 22：完成 `Memora v0.19`，替通過 Policy 的記憶加入 1～5 分 Importance Score，並以 Relevance + Importance 重新排序。
- Day 23：完成 `Memora v0.20`，以 `last_accessed_at` 與七天 Half-life 將 Recency 納入 Retrieval，並加入可控的 `forget` 指令。
- Day 24：完成 `Memora v0.21`，在寫入前搜尋相近記憶，區分 create、skip、update、keep_both 與 review，維護長期記憶的一致性與歷史意義。

## 執行方式

需求：Python 3.10 以上，以及可使用 OpenAI API 的金鑰。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell：.venv\Scripts\Activate.ps1
pip install -r requirements.txt
export OPENAI_API_KEY="你的 API Key"
python chatbot.py
```

輸入 `history` 可查看目前對話，`memories` 可查看長期記憶及 Importance／Recency，`policy` 可查看最近一次寫入判斷，`search <query>` 可測試語意搜尋，`forget <memory_id>` 可依完整 ID 刪除指定的長期記憶，`profile` 可查看目前使用者設定，`exit` 則結束程式。手動記憶格式為 `remember semantic <內容>` 或 `remember episodic <內容>`。API Key 由環境變數讀取，不會寫進原始碼或 Git repository。

Conversation History 只存在目前 Process；經過抽取的 Long-term Memory 會保存在 `memora_db/`，User Profile 則保存在 `user_profile.json`。兩者都已排除於 Git，以免誤提交個人資料。

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
- [Day 15｜不用 Vector Database，自己實作一次 Semantic Search](https://ithelp.ithome.com.tw/articles/10407997)
- [Day 16｜Vector Database 到底在做什麼？](https://ithelp.ithome.com.tw/articles/10408000)
- [Day 17｜打造第一個 Long-term Memory Store](https://ithelp.ithome.com.tw/articles/10408001)
- [Day 18｜讓 AI 找回過去：Memory Retrieval](https://ithelp.ithome.com.tw/articles/10408560)
- [Day 19｜Memory ≠ User Profile：記住事情和認識一個人的差別](https://ithelp.ithome.com.tw/articles/10409065)
- [Day 20｜Semantic Memory vs Episodic Memory：AI 到底記得什麼？](https://ithelp.ithome.com.tw/articles/10409278)
- [Day 21｜AI 應該什麼都記住嗎？Memory Policy](https://ithelp.ithome.com.tw/articles/10409577)
- [Day 22｜Importance Score：哪些記憶比較重要？](https://ithelp.ithome.com.tw/articles/10409812)
- [Day 23｜AI 也需要遺忘：Memory Decay、Recency 與 Forgetting](https://ithelp.ithome.com.tw/articles/10410230)
- [Day 24｜AI 記錯了怎麼辦？Deduplication、Update 與 Contradiction](https://ithelp.ithome.com.tw/articles/10410912)
