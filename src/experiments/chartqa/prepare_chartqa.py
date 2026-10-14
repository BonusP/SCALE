"""ChartQA 공식 배포본에서 실험 입력(PNG·GT 표·업로드용 PDF)을 준비합니다.

    python experiments/chartqa/prepare_chartqa.py --chartqa "ChartQA Dataset.zip"

`--chartqa`에는 내려받은 zip 또는 압축을 푼 `ChartQA Dataset` 폴더를 줍니다.
ids.txt 순서대로 test split의 png/tables를 복사하고, 한 페이지에 차트 한 장씩 담은
selected_images.pdf를 만듭니다. PDF는 실험 당시와 같은 방식(matplotlib 3.10.9,
dpi 100, interpolation="none")으로 만들어 원본 PNG 픽셀이 손실 없이 들어갑니다.
"""
import argparse
import io
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def reader(source: Path):
    """split/kind/name 경로를 받아 바이트를 돌려주는 함수를 만듭니다."""
    if source.is_file():
        archive = zipfile.ZipFile(source)
        return lambda rel: archive.read(f"ChartQA Dataset/{rel}")
    root = source / "ChartQA Dataset" if (source / "ChartQA Dataset").is_dir() else source
    return lambda rel: (root / rel).read_bytes()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chartqa", required=True, type=Path, help="ChartQA Dataset.zip 또는 압축을 푼 폴더")
    ap.add_argument("--ids", type=Path, default=HERE / "ids.txt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "chartqa")
    a = ap.parse_args()

    read = reader(a.chartqa)
    ids = [line.strip() for line in a.ids.read_text(encoding="utf-8").splitlines() if line.strip()]
    (a.out / "png").mkdir(parents=True, exist_ok=True)
    (a.out / "tables").mkdir(parents=True, exist_ok=True)
    pdf_path = a.out / "selected_images.pdf"
    with PdfPages(pdf_path) as pdf:
        for chart_id in ids:
            png = read(f"{a.split}/png/{chart_id}.png")
            (a.out / "png" / f"{chart_id}.png").write_bytes(png)
            (a.out / "tables" / f"{chart_id}.csv").write_bytes(read(f"{a.split}/tables/{chart_id}.csv"))
            image = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
            height, width = image.shape[:2]
            figure = plt.figure(figsize=(width / 100, height / 100), dpi=100)
            axes = figure.add_axes([0, 0, 1, 1])
            axes.imshow(image, interpolation="none")
            axes.axis("off")
            pdf.savefig(figure)
            plt.close(figure)
    print(f"{len(ids)} charts -> {a.out}")
    print(f"Upload this PDF: {pdf_path}")


if __name__ == "__main__":
    main()
