"""Download public Qwen weights from the Qwen namespace on ModelScope; verify SHA256."""
from __future__ import annotations
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DESTINATION = ROOT / "models/Qwen2.5-1.5B-Instruct"

def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()

def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest_path = DESTINATION / "download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        api = f"https://modelscope.cn/api/v1/models/{MODEL}/repo/files?Revision=master&Recursive=true"
        with urllib.request.urlopen(api, timeout=60) as response:
            metadata = json.load(response)
        if not metadata.get("Success"):
            raise RuntimeError("ModelScope metadata request failed")
        names = {"LICENSE", "README.md", "config.json", "generation_config.json", "merges.txt", "vocab.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"}
        files = [item for item in metadata["Data"]["Files"] if item["Name"] in names]
        assert {item["Name"] for item in files} == names
        manifest = {"model_id": MODEL, "source": "https://modelscope.cn/models/" + MODEL, "files": files}
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for item in sorted(manifest["files"], key=lambda entry: entry["Size"]):
        target = DESTINATION / item["Name"]
        if target.exists() and target.stat().st_size == item["Size"] and digest(target) == item["Sha256"]:
            print("Verified existing", target.name, flush=True)
            continue
        partial = target.with_suffix(target.suffix + ".part")
        if partial.exists() and partial.stat().st_size == item["Size"] and digest(partial) == item["Sha256"]:
            partial.replace(target)
            print("Verified complete partial", target.name, flush=True)
            continue
        url = f"https://modelscope.cn/models/{MODEL}/resolve/{item['Revision']}/{urllib.parse.quote(item['Name'])}"
        for attempt in range(4):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {"User-Agent": "tax-extraction-poc/1.0"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                    append = offset > 0 and response.status == 206 and response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                    done = offset if append else 0
                    last_report = done
                    with partial.open("ab" if append else "wb") as output:
                        while True:
                            block = response.read(4 * 1024 * 1024)
                            if not block:
                                break
                            output.write(block)
                            done += len(block)
                            if done - last_report >= 128 * 1024 * 1024:
                                print(f"{target.name}: {done / 2**20:.0f}/{item['Size']/2**20:.0f} MiB", flush=True)
                                last_report = done
                if partial.stat().st_size != item["Size"] or digest(partial) != item["Sha256"]:
                    raise RuntimeError(f"File integrity verification failed: {target.name}")
                partial.replace(target)
                print("Downloaded and verified", target.name, flush=True)
                break
            except Exception as error:
                print(f"Retry {attempt+1}/4 {target.name}: {type(error).__name__}: {error}", flush=True)
                if attempt == 3:
                    raise
                time.sleep(3)
    print("MODEL_READY", str(DESTINATION), flush=True)

if __name__ == "__main__":
    main()
