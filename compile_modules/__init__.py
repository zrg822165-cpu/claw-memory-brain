"""Compile modules package.

当前主链路已切到 event chunk first：
- `event_chunker.py`：调用云端 LLM 做事件切分
- `event_filter.py`：做轻量硬过滤
- `event_store.py`：写入 canonical event-chunk store
- `indexer.py`：从 SQLite 主表派生 High LanceDB 向量索引
"""
