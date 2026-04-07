# Codex Agent Instructions — whisper-guard
<!-- AGENTS.md v2 | 2026-04-07 | 審計升級：編碼規則 + 交接路徑 + git init -->

## 任務來源
實作規格在 `CODEX_HANDOVER.md` — 開始前先讀完，依規格執行。

## Environment Constraints
- **語法相容 Python 3.9+** — 禁用 match/case、`str | None`（用 `Optional[str]`）、`list[dict]`（用 `List[dict]`）等 3.10+ 語法。**測試可以用任何 >= 3.9 的版本跑**（本機是 3.12，沒問題）
- **零外部依賴** — 只用 stdlib（`re`, `dataclasses`, `typing`）
- **測試框架**: pytest（如未安裝：`pip install pytest`）

## Debugging Protocol
1. **Root Cause 優先** — 沒找到原因就不准動手修
2. **Pattern Analysis** — 找 codebase 中同類正常範例比對
3. **三振出局** — 同一問題 3 次不同方法仍失敗 → 停下來，輸出已嘗試方法 + 建議下一步
4. **只修 root cause** — 不用 workaround 掩蓋

## Verification Gate
宣稱「完成」前必須通過：
1. `pip install -e .` 成功
2. `python -c "from whisper_guard import WhisperGuard; print('OK')"` 成功
3. `pytest tests/ -v` 全綠
4. 指出哪個指令/輸出證明宣稱 — 禁止用「should work」「probably fixed」「Done!」

## 輸出風格
- 簡潔直接，先結論後理由
- 用表格做比較和摘要
- commit message 寫「為什麼」不只是「什麼」

## 編碼規則
- **所有檔案必須是 UTF-8** — commit 前確認中文可讀，不能有亂碼
- 如果工具輸出亂碼，**停下來回報**，不要把亂碼寫進檔案

## Git 初始化
這是新專案，如果尚未有 `.git/`：
```bash
git init && git add -A && git commit -m "init: whisper-guard v0.1.0 scaffold"
```
不要推到 remote — Hevin 會手動建 GitHub repo。

## 交接路徑（必做，不可跳過）
完成任務後，**必須**在專案根目錄建立 `CODEX_RESULT.md`，這是 CC 審計的入口。缺少此檔案視為交付不完整。

內容：
1. 完成了什麼（逐項對照 CODEX_HANDOVER.md 的 Pending，用 checkbox）
2. 測試結果（完整 pytest 輸出，不是摘要）
3. 有疑慮的項目（`⚠️ REVIEW:` 標記）
4. 未完成的項目（如有）
5. 與 spec 不一致的實作決策（如有，說明為什麼偏離）

## 審計協定
- 你的產出會經過 Claude Code 審計，不會直接合併
- commit 前跑完 Verification Gate，附上測試結果
- 有疑慮的實作用 `⚠️ REVIEW:` 標記，不要藏
- 審計發現重大缺失 → 此 AGENTS.md 會被更新補強規則

## Don't
- 不要發 PyPI
- 不要建 GitHub repo
- 不要加 CI/CD、pre-commit、mypy
- 不要加不必要的功能、註解、docstring、type annotation
- 不要用 emoji 除非被要求
