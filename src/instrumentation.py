import json, os
from datetime import datetime

LOG_FILE = "/media/pi/KINGSTON/local_ai/logs/queries.jsonl"


def log_query(record):
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        record["timestamp"] = datetime.now().isoformat()
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[instrumentation] log failed: {e}")


def new_record(query, path="model_only"):
    return {
        "query": query,
        "path": path,
        "rag_used": False,
        "sources": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "time_to_first_token": 0.0,
        "decode_time": 0.0,
        "total_time": 0.0,
        "tokens_per_second": 0.0,
        "model": "",
        "query_type": "",
        "skip_retrieval": False,
        "confidence": 0.0,
        "search_time": 0.0,
        "num_results": 0,
        "tiers_used": [],
    }
