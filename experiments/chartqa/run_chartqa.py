"""ChartQA PDF를 실행 중인 SCALE 서버에 올려 추출하고, 끝나면 RMS-F1로 채점합니다.

    python experiments/chartqa/run_chartqa.py

서버(uvicorn)가 먼저 실행 중이어야 합니다. 이미 끝난 job을 다시 채점하려면 --job <job_id>를 줍니다.
"""
import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from evaluate import evaluate  # noqa: E402
from prepare_eval import DONE_STATUSES, prepare  # noqa: E402


def _request(url: str, data: bytes | None = None, headers: dict | None = None) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers or {})) as response:
        return json.loads(response.read().decode("utf-8"))


def upload(server: str, pdf: Path) -> str:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{pdf.name}\"\r\n"
            "Content-Type: application/pdf\r\n\r\n").encode() + pdf.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    payload = _request(f"{server}/api/jobs", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return payload["job_id"]


def wait(server: str, job_id: str, interval: float) -> dict:
    last = None
    while True:
        job = _request(f"{server}/api/jobs/{job_id}")["job"]
        state = (job["status"], job.get("phase"), job.get("progress"))
        if state != last:
            print(f"  {job['status']:<22} {job.get('progress') or 0:>3}%  {job.get('phase') or ''}", flush=True)
            last = state
        if job["status"] in DONE_STATUSES:
            return job
        if job["status"] == "failed":
            raise SystemExit(f"job failed: {job.get('error')}")
        time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8092")
    ap.add_argument("--pdf", type=Path, default=REPO / "data" / "chartqa" / "selected_images.pdf")
    ap.add_argument("--job", help="이미 끝난 job을 채점만 할 때 지정")
    ap.add_argument("--interval", type=float, default=10.0, help="진행 상황 확인 간격(초)")
    a = ap.parse_args()

    if a.job:
        job_id = a.job
    else:
        job_id = upload(a.server, a.pdf)
        print(f"Uploaded {a.pdf.name} -> job {job_id}")
    wait(a.server, job_id, a.interval)
    run = prepare(job_id)
    print(json.dumps(evaluate(run, HERE / "runs" / run / "dataset"), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
