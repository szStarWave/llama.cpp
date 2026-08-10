import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "herdsman-cpu"
CLIENT = OUT / "herdsman-llamacpp-e2e.test.exe"
IMAGE = ROOT / "validation-logs" / "b10068-nonturbo-20260809" / "vision-fixture.png"
MODEL = Path(r"C:\herdsman\models\Qwen3.5-27B\windows\amd64\amd\Qwen3.5-27B-Q4_K_M.gguf")
MMPROJ = Path(r"C:\herdsman\models\Qwen3.5-27B\windows\amd64\amd\mmproj-BF16.gguf")
SERVER = ROOT / "build-validation-b10068" / "bin" / "Release" / "llama-server.exe"
FATAL_MARKERS = (
    "access violation",
    "exception code: 0xc0000005",
    "ggml_abort",
    "assertion failed",
    "nan detected",
)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ServerProcess:
    def __init__(self, kv_type, phase):
        self.kv_type = kv_type
        self.phase = phase
        self.port = free_port()
        self.proc = None
        self.stdout = None
        self.stderr = None
        stem = f"{kv_type}-{phase}"
        self.stdout_path = OUT / f"{stem}.server.stdout.log"
        self.stderr_path = OUT / f"{stem}.server.stderr.log"

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        args = [
            str(SERVER),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--no-webui",
            "--verbose",
            "--jinja",
            "--ssd-kv-offload-gb", "4",
            "--dram-kv-offload-gb", "2",
            "--kv-cache-resume-policy", "1",
            "--aidaptiv-cache-prefix", f"herdsman-t34-{self.kv_type}-",
            "-m", str(MODEL),
            "--mmproj", str(MMPROJ),
            "--ctx-size", "8192",
            "--cache-ram", "0",
            "--device", "none",
            "--gpu-layers", "0",
            "--threads", "4",
            "--threads-batch", "4",
            "-np", "1",
            "--batch-size", "256",
            "--ubatch-size", "128",
            "--cache-type-k", "q8_0",
            "--cache-type-v", self.kv_type,
            "--no-mmap",
            "--no-warmup",
            "--temp", "0.7",
            "--top-p", "0.95",
            "--top-k", "40",
            "--repeat-penalty", "1.1",
        ]
        self.stdout = self.stdout_path.open("w", encoding="utf-8")
        self.stderr = self.stderr_path.open("w", encoding="utf-8")
        self.stdout.write("COMMAND " + subprocess.list2cmdline(args) + "\n")
        self.stdout.flush()
        self.proc = subprocess.Popen(
            args,
            cwd=SERVER.parent,
            stdout=self.stdout,
            stderr=self.stderr,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
        )
        deadline = time.time() + 1200
        while time.time() < deadline:
            code = self.proc.poll()
            if code is not None:
                raise RuntimeError(f"server exited during startup: {code}")
            try:
                response = requests.get(self.base_url + "/props", timeout=2)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError("server startup timed out")

    def stop(self):
        if self.proc is None:
            return None
        if self.proc.poll() is None:
            try:
                requests.post(self.base_url + "/shutdown", json={}, timeout=10)
            except requests.RequestException:
                pass
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(timeout=30)
        code = self.proc.returncode
        if self.stdout:
            self.stdout.close()
        if self.stderr:
            self.stderr.close()
        return code

    def fatal_log_markers(self):
        contents = ""
        for path in (self.stdout_path, self.stderr_path):
            if path.exists():
                contents += path.read_text(encoding="utf-8", errors="replace").lower()
        return [marker for marker in FATAL_MARKERS if marker in contents]


def run_client(server, scenarios):
    env = os.environ.copy()
    env["HERDSMAN_E2E_BASE_URL"] = server.base_url
    env["HERDSMAN_E2E_IMAGE"] = str(IMAGE)
    env["HERDSMAN_E2E_SCENARIOS"] = ",".join(scenarios)
    command = [str(CLIENT), "-test.run", "^TestHerdsmanLlamaCppE2E$", "-test.v"]
    return subprocess.run(
        command,
        cwd=OUT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=3600,
    )


def run_phase(kv_type, phase, scenarios):
    result = {
        "kv": f"q8_0/{kv_type}",
        "phase": phase,
        "scenarios": scenarios,
        "passed": False,
    }
    server = ServerProcess(kv_type, phase)
    client_result = None
    try:
        server.start()
        client_result = run_client(server, scenarios)
        result["client_exit"] = client_result.returncode
        result["client_stdout"] = client_result.stdout
        result["client_stderr"] = client_result.stderr
        result["passed"] = client_result.returncode == 0 and server.proc.poll() is None
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        result["server_exit"] = server.stop()
        result["fatal_log_markers"] = server.fatal_log_markers()
        result["passed"] = (
            result["passed"]
            and result["server_exit"] == 0
            and not result["fatal_log_markers"]
        )
        if client_result is not None:
            stem = f"{kv_type}-{phase}"
            (OUT / f"{stem}.client.stdout.log").write_text(client_result.stdout, encoding="utf-8")
            (OUT / f"{stem}.client.stderr.log").write_text(client_result.stderr, encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("client_std")}), flush=True)
    return result


def validate_inputs():
    missing = [path for path in (CLIENT, IMAGE, MODEL, MMPROJ, SERVER) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing test inputs: " + ", ".join(str(path) for path in missing))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    validate_inputs()
    selected = sys.argv[1:] or ["turbo3", "turbo4"]
    unknown = set(selected) - {"turbo3", "turbo4"}
    if unknown:
        raise ValueError(f"unsupported KV types: {sorted(unknown)}")

    results = []
    results_path = OUT / ("results-" + "-".join(selected) + ".json")
    for kv_type in selected:
        phases = [
            ("first", ["text", "document-image", "single-image", "two-turn-two-image"]),
            ("restart", ["text", "document-image"]),
        ]
        for phase, scenarios in phases:
            results.append(run_phase(kv_type, phase, scenarios))
            results_path.write_text(
                json.dumps(results, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
    if not all(item["passed"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
