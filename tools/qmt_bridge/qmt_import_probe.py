from datetime import datetime

try:
    from xtquant import xtdata
    message = "OK " + str(datetime.now()) + " " + str(xtdata)
except Exception as exc:
    message = "ERR " + repr(exc)

with open(
    r"D:\chatgpt\aiagents-stock\tick-stock-panel-v022-integrated\data\import_reports\qmt_probe.log",
    "a",
    encoding="utf-8",
) as handle:
    handle.write(message + "\n")
