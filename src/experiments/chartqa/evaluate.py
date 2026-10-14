"""예측 테이블 vs ChartQA 정답 테이블 채점 (RMS-F1).

    python experiments/chartqa/evaluate.py [--run chartqa]

RMS(Relative Mapping Similarity, DePlot: Liu et al., 2022, arXiv:2212.10505 §3.1):
표를 (행 헤더 r, 열 헤더 c, 값 v) 튜플 집합으로 보고
    키 유사도   = 1 - NL_tau(p_r||p_c, t_r||t_c)         (정규화 Levenshtein, tau 초과는 1로 절단)
    튜플 유사도 = (1 - NL_tau) * (1 - D_theta(p_v, t_v))   (D_theta: 상대오차가 theta 이상이면 1)
키 유사도로 헝가리안 매칭 -> 매칭된 튜플 유사도의 합 / N (precision), / M (recall) -> F1.
표와 그 전치본 중 F1이 높은 쪽을 채택한다(행/열 순열·전치 불변).
논문은 tau, theta의 구체 값을 명시하지 않아 기본 1.0(절단 없음)이고 옵션으로 바꿀 수 있다.
매칭되지 않은 튜플은 유사도 0으로 계산(= 비용 1)한다.
"""
import argparse
import csv
import json
import re
import statistics

import numpy as np
from pathlib import Path

from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
TAU = THETA = 1.0  # RMS 절단 임계값 (main의 --tau/--theta 로 변경)
Entry = tuple[str, str, float]  # (row header, column header, value)


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def nl(a: str, b: str, tau: float = 1.0) -> float:
    """정규화 Levenshtein 거리, tau 이상(초과)이면 1."""
    m = max(len(a), len(b))
    d = _lev(a, b) / m if m else 0.0
    return d if d <= tau else 1.0 if tau < 1.0 else d

def rel_dist(p: float, t: float) -> float:
    if t == 0:
        return 0.0 if p == 0 else 1.0
    return min(1.0, abs(p - t) / abs(t))

def _rms_once(pred: list[Entry], gt: list[Entry], tau: float, theta: float):
    n, m = len(pred), len(gt)

    def key(e):
        return f"{_norm(e[0])}||{_norm(e[1])}"

    key_sim = np.zeros((n, m))
    for i, p in enumerate(pred):
        for j, t in enumerate(gt):
            key_sim[i, j] = 1 - nl(key(p), key(t), tau)
    r, c = linear_sum_assignment(-key_sim)  # 키 유사도 최대 매칭
    total = 0.0
    for i, j in zip(r, c):
        d = rel_dist(pred[i][2], gt[j][2])
        d = d if d < theta else 1.0
        total += key_sim[i, j] * (1 - d)
    prec, rec = total / n, total / m
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1

def rms(pred: list[Entry], gt: list[Entry], tau: float = 1.0, theta: float = 1.0):
    """(precision, recall, f1)."""
    if not pred or not gt:
        return 0.0, 0.0, 0.0
    a = _rms_once(pred, gt, tau, theta)
    tr = [(c, r, v) for r, c, v in pred]
    b = _rms_once(tr, gt, tau, theta)
    return max(a, b, key=lambda x: x[2])

def to_num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if v == v else None
    s = str(v).strip().replace(",", "").rstrip("%").replace("$", "")
    try:
        x = float(s)
        return x if x == x else None
    except ValueError:
        return None

def load_gt(path: Path) -> dict[str, dict[str, float]]:
    rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
    header, body = rows[0], rows[1:]
    gt = {}
    for ci, name in enumerate(header[1:], start=1):
        cells = {}
        for r in body:
            if len(r) > ci and (v := to_num(r[ci])) is not None:
                cells.setdefault(r[0], v)
        if cells:  # 값이 전부 nan인 열은 정답 시리즈로 치지 않음
            gt[name or f"col{ci}"] = cells
    return gt

def load_pred(rec: dict) -> dict[str, dict[str, float]]:
    """예측 시리즈 로드. 같은 범례 이름의 곡선 조각(프로그램이 구간별 세그먼트로 나눠 내는 경우)은 하나로 합친다."""
    pred, dense_by_label = {}, {}
    rows = rec["series_data"].get("series") or []
    for i, s in enumerate(rows):
        label = s.get("legend_label") or f"series_{i}"
        # 시리즈·조각 구분: 같은 범례의 전압·전력 및 segment 경계를 보존합니다.
        axes = {row.get("axis_binding") for row in rows if row.get("legend_label") == s.get("legend_label")}
        if s.get("segments") and len(axes) > 1:
            label = f"{label} [{s.get('axis_binding')}]"
        if s.get("segments"):
            segments = [[(float(x), v) for x, y in segment.get("points_data", [])
                         if not isinstance(x, str) and (v := to_num(y)) is not None]
                        for segment in s["segments"]]
            pred[label] = DenseCurve([], segments=[points for points in segments if points])
            continue
        cells, dense = {}, []
        for x, y in s.get("points_data") or []:
            if (v := to_num(y)) is not None:
                if isinstance(x, str):
                    cells.setdefault(x, v)
                else:
                    dense.append((float(x), v))
        if dense:  # 숫자 x축(픽셀 추적) 곡선: 정답 x에서 보간해 채점 (resample_dense)
            dense_by_label.setdefault(label, []).extend(dense)
        elif cells:
            pred[label] = cells
    for label, pts in dense_by_label.items():
        pred[label] = DenseCurve(pts)
    return pred

class DenseCurve(dict):
    """숫자 x의 밀집 예측 곡선. 정답 x 위치에서 선형보간해 {정답 카테고리: 값} 로 변환한다."""

    def __init__(self, pts, segments=None):
        super().__init__()
        self.pts = sorted(pts)
        self.segments = [sorted(points) for points in segments] if segments is not None else None

def resample_dense(pred, gt):
    """DenseCurve 를 정답 카테고리(숫자로 파싱 가능한 것)의 x에서 보간. 예측 x 범위 밖은 외삽하지 않고 누락."""
    gt_cats = {c for cells in gt.values() for c in cells}
    xs = {c: to_num(c) for c in gt_cats}
    out = {}
    for name, cells in pred.items():
        if not isinstance(cells, DenseCurve):
            out[name] = cells
            continue
        res = {}
        # 시리즈·조각 구분: segment 사이의 빈 구간은 보간하지 않습니다.
        for points in cells.segments if cells.segments is not None else [cells.pts]:
            if not points:
                continue
            px, py = zip(*points)
            lo, hi = px[0], px[-1]
            pad = 0 if cells.segments is not None else 0.005 * (hi - lo)
            for c, x in xs.items():
                if x is not None and lo - pad <= x <= hi + pad:
                    res.setdefault(c, float(np.interp(x, px, py)))
        if res:
            out[name] = res
    return out

def gt_entries(gt):
    return [(cat, col, v) for col, cells in gt.items() for cat, v in cells.items()]

def pred_entries(rec, pred):
    """예측 -> (row, col, value). group_label 없는 단일 시리즈는 열 헤더로 value_axis_label을 쓴다."""
    axis = str((rec.get("table") or {}).get("value_axis_label") or "")
    multi = len(pred) > 1
    return [(cat, col if multi or col not in ("values",) else axis, v)
            for col, cells in pred.items() for cat, v in cells.items()]


def evaluate(run: str, data: Path) -> dict:
    run_dir = HERE / "runs" / run
    rows = []
    for line in (data / "manifest.jsonl").read_text().splitlines():
        it = json.loads(line)
        raw = run_dir / "raw" / f"{it['id']}.json"
        if not raw.exists():
            continue
        rec = json.loads(raw.read_text())
        gt = load_gt(data / it["table"])
        pred = {} if rec["failed"] else resample_dense(load_pred(rec), gt)
        _, _, f1 = rms(pred_entries(rec, pred), gt_entries(gt), TAU, THETA)
        rows.append({"id": it["id"], "status": rec.get("status", "extracted" if pred else "empty"), "rms_f1": f1})

    with (run_dir / "per_image.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {"run": run, "RMS_F1": round(statistics.mean(r["rms_f1"] for r in rows), 4),
               "n_images": len(rows), "tau": TAU, "theta": THETA}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="chartqa", help="runs/ 아래 이름 (prepare_eval.py의 --name)")
    ap.add_argument("--data", help="manifest.jsonl과 GT 표가 있는 폴더 (기본: runs/<run>/dataset)")
    ap.add_argument("--tau", type=float, default=1.0, help="RMS 텍스트 거리 절단값")
    ap.add_argument("--theta", type=float, default=1.0, help="RMS 숫자 상대오차 절단값")
    a = ap.parse_args()
    global TAU, THETA
    TAU, THETA = a.tau, a.theta
    data = Path(a.data) if a.data else HERE / "runs" / a.run / "dataset"
    print(json.dumps(evaluate(a.run, data), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
