"""Run remotely. Fetch a pinned official implementation and apply mechanical portability edits."""
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "f7b908b5f8f23bc0462393c14d041354985af3bd"
ROOT = Path("external/dst_sgnn")
FILES = ("MODEL.py", "SGSC.py", "RevIN.py")


def main():
    target = ROOT / "dst_sgnn_upstream"
    target.mkdir(parents=True, exist_ok=True)
    (target / "__init__.py").write_text("", encoding="utf-8")
    manifest = {"repository": "https://github.com/komorebi424/DST-SGNN", "revision": REVISION,
                "original_sha256": {}, "adapted_sha256": {},
                "changes": ["package-relative imports", "remove forced cuda:0 placement",
                            "use float32 rather than mixed float64/float32"]}
    for filename in FILES:
        url = f"https://raw.githubusercontent.com/komorebi424/DST-SGNN/{REVISION}/model/{filename}"
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
        manifest["original_sha256"][filename] = hashlib.sha256(raw).hexdigest()
        text = raw.decode().replace("from model.", "from dst_sgnn_upstream.")
        text = text.replace("self.cuda(0)", "pass").replace("self.to('cuda:0')", "pass")
        text = text.replace(".double()", ".float()")
        output = target / filename
        output.write_text(text, encoding="utf-8", newline="\n")
        manifest["adapted_sha256"][f"dst_sgnn_upstream/{filename}"] = hashlib.sha256(output.read_bytes()).hexdigest()
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
