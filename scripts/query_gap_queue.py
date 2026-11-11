import json
import sqlite3

conn = sqlite3.connect("file:/app/data/user_data/minute_gap_queue.sqlite?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
print(json.dumps({
    "status": [dict(row) for row in conn.execute(
        "select status,count(*) n,min(trade_date) first,max(trade_date) last "
        "from minute_gap_queue group by status order by status"
    )],
    "candidate_dates": [dict(row) for row in conn.execute(
        "select trade_date,count(*) n from minute_gap_queue "
        "where status in ('pending','retry','qmt_turnover_conflict','source_incomplete',"
        "'source_empty','source_unavailable','quarantine','source_no_record',"
        "'existing_values_conflict','daily_baseline_repair_candidate') "
        "and trade_date >= '2024-01-01' group by trade_date order by trade_date"
    )],
}, ensure_ascii=False, indent=2))
