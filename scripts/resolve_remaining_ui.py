"""Apply reviewed UI conflict resolutions, retaining nonconflicting custom edits."""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
artifacts = Path(r"F:\tick-panel-upgrade-20261009\integration\conflicts")
pattern = r"<<<<<<<[^\n]*\n(.*?)=======\n(.*?)>>>>>>>[^\n]*\n"
files = ["frontend/src/pages/Watchlist.tsx", "frontend/src/pages/Factors.tsx",
         "frontend/src/pages/backtest/MiningWorkbench.tsx"]
for name in files:
    def choose(match):
        local, upstream = match.groups()
        if name.endswith("Watchlist.tsx"):
            if "lucide-react" in upstream:
                return upstream.replace("FileUp,", "FileUp, Sparkles, Loader2,")
            return local + upstream
        return upstream
    source = (artifacts / name).read_text(encoding="utf-8")
    result, count = re.subn(pattern, choose, source, flags=re.S)
    assert count > 0 and "<<<<<<<" not in result
    (root / name).write_text(result, encoding="utf-8")
    print(name, count)
