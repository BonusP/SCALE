"""완료된 추출 job의 결과를 evaluate.py 입력 형식으로 변환합니다.

    python experiments/chartqa/prepare_eval.py [--job <job_id>]

--job을 생략하면 selected_images.pdf를 처리한 job 중 가장 최근에 완료된 것을 씁니다.

PDF의 n번째 페이지(chart_id의 `_img{n:03d}_`)를 ids.txt의 n번째 ChartQA id와 짝짓습니다.
OECD 표는 행이 series·열이 연도이므로 evaluate.py가 기대하는 방향(행=x, 열=series)으로 전치합니다.
결과는 runs/<name>/ (기본: runs/chartqa/) 에 raw/, dataset/, mapping.json, job.json 으로 저장됩니다.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PDF_NAME = "selected_images.pdf"
DONE_STATUSES = ("completed", "completed_with_skips")


def prepare(job_id: str | None = None, data_root: Path = REPO / "data",
            gt_dir: Path = REPO / "data" / "chartqa" / "tables", ids_path: Path = HERE / "ids.txt",
            name: str = "chartqa") -> str:
    """Write runs/<name>/ for evaluate.py and return the run name."""
    a = argparse.Namespace(job=job_id, data_root=data_root, gt_dir=gt_dir, ids=ids_path, name=name)
    ids = [line.strip() for line in a.ids.read_text(encoding="utf-8").splitlines() if line.strip()]
    page_gt = {index: gt_id for index, gt_id in enumerate(ids, start=1)}
    run = HERE / "runs" / a.name
    raw = run / "raw"
    dataset = run / "dataset"
    raw.mkdir(parents=True, exist_ok=True)
    dataset.mkdir(exist_ok=True)
    with sqlite3.connect((a.data_root / "mvp.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as c:
        c.row_factory = sqlite3.Row
        if a.job:
            row = c.execute("select * from jobs where job_id=?", (a.job,)).fetchone()
            if row is None:
                raise SystemExit(f"job not found: {a.job}")
        else:
            row = c.execute(
                "select * from jobs where source_pdf_name=? and status in (?, ?) order by created_at desc limit 1",
                (PDF_NAME, *DONE_STATUSES),
            ).fetchone()
            if row is None:
                raise SystemExit(f"no completed job for {PDF_NAME}; upload it and wait until the job is completed")
        job = dict(row)
        a.job = job["job_id"]
        print("Job:", a.job, f"({job['status']})")
        charts = [dict(r) for r in c.execute("select * from charts where job_id=? order by image_name", (a.job,))]
    manifest, seen = [], set()
    for chart in charts:
        page = int(re.search(r"_img(\d+)_", chart["chart_id"]).group(1))
        assert page in page_gt and page not in seen, chart["chart_id"]
        seen.add(page)
        gt_id = page_gt[page]
        source_gt = a.gt_dir / (gt_id + ".csv")
        source_bytes = source_gt.read_bytes()
        rows = list(csv.reader(source_bytes.decode("utf-8-sig").splitlines()))
        # OECD tables list series as rows and years as columns; evaluate.py expects the reverse.
        transposed = rows[0][0] == "Country"
        if transposed:
            assert len({len(r) for r in rows}) == 1
            rows = [list(r) for r in zip(*rows)]
            rows[0][0] = "Characteristic"
        stream = io.StringIO(newline="")
        csv.writer(stream).writerows(rows)
        (dataset / (gt_id + ".csv")).write_text(stream.getvalue(), encoding="utf-8")
        # Read results from the job folder directly, independent of the absolute path stored in SQLite.
        result = a.data_root / "jobs" / a.job / "plotlogic_runs" / chart["chart_id"]
        series_path = result / "curve_series_data.json"
        series = json.loads(series_path.read_text(encoding="utf-8")) if series_path.exists() else {"series": []}
        has_data = any(s.get("points_data") for s in series.get("series", []))
        pred_labels = sorted({str(s.get("legend_label") or "").strip() for s in series.get("series", [])})
        gt_labels = [s.strip() for s in rows[0][1:]]
        table_path = result / "codex_categorical_table.json"
        record = {"series_data": series,
                  "table": json.loads(table_path.read_text(encoding="utf-8")) if table_path.exists() else {},
                  "failed": chart["status"] != "extracted" or not has_data,
                  "status": chart["status"] if has_data else "empty", "elapsed_sec": None,
                  "job_id": a.job, "chart_id": chart["chart_id"], "source_result_dir": str(result),
                  "extraction_family": chart["extraction_family"]}
        (raw / (gt_id + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append({"id": gt_id, "table": gt_id + ".csv", "page": page, "chart_id": chart["chart_id"],
                         "image_name": chart["image_name"], "result_dir": str(result),
                         "extraction_family": chart["extraction_family"],
                         "gt_source": str(source_gt), "gt_sha256": hashlib.sha256(source_bytes).hexdigest(),
                         "gt_transposed_for_evaluator": transposed,
                         "gt_series": gt_labels, "pred_series": pred_labels,
                         "series_labels_match": {s.lower() for s in gt_labels} == {s.lower() for s in pred_labels}})
    assert seen == set(page_gt), f"missing pages: {sorted(set(page_gt) - seen)}"
    (dataset / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest),
                                            encoding="utf-8")
    (run / "mapping.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Prepared:", len(manifest), "Run:", run.name)
    return run.name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", help=f"{PDF_NAME}를 추출한 job_id (기본: 가장 최근에 완료된 job)")
    ap.add_argument("--data-root", type=Path, default=REPO / "data", help="mvp.sqlite3와 jobs/가 있는 폴더")
    ap.add_argument("--gt-dir", type=Path, default=REPO / "data" / "chartqa" / "tables")
    ap.add_argument("--ids", type=Path, default=HERE / "ids.txt")
    ap.add_argument("--name", default="chartqa", help="runs/ 아래 이름")
    a = ap.parse_args()
    run = prepare(a.job, a.data_root, a.gt_dir, a.ids, a.name)
    print(f"Next: python experiments/chartqa/evaluate.py --run {run}")


if __name__ == "__main__":
    main()
