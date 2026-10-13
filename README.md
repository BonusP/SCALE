# SCALE

Official prototype of **SCALE: An Automated Framework for Numerical Data Reconstruction from Complex Scientific Charts**.

SCALE takes a scientific paper PDF, finds its line charts, and reconstructs the numerical data of every curve as CSV. A vision-language model (via Codex CLI) classifies panels and reads axes and legends. A pixel-level pipeline then traces each curve and maps it to data coordinates.

## Pipeline

The backend runs four sequential modules:

1. **Chart Image Extraction**: collects figure images from the input PDF (`opendataloader-pdf` + Poppler), identifies their panels, and selects target CMSL charts with the VLM.
2. **Chart Element Extraction**: detects plot regions, axes, ticks, and legends with the VLM, refined by OCR (RapidOCR).
3. **Curve Extraction**: identifies individual curves by color and extracts their pixel coordinates.
4. **Numerical Value Extraction**: converts pixel coordinates into numerical values using axis and tick information. On dual-y charts, the VLM assigns each curve to its axis.

Extracted chart elements and curves are shown in the web UI, and numerical data are exported with the **Download CSV** button.

| Component | Technology |
| --- | --- |
| Frontend | HTML, CSS, vanilla JavaScript |
| Backend | FastAPI, Uvicorn (JSON-based HTTP API) |
| VLM | GPT-5.5 (`gpt-5.5`) via Codex CLI |
| Image processing | Pillow, NumPy |
| Storage | Local file system (results), SQLite (job metadata) |

## Requirements

| Requirement | Required | Notes |
| --- | --- | --- |
| Python 3.12 | Yes | `rapidocr-onnxruntime==1.4.4` does not support 3.13+ |
| [Codex CLI](https://github.com/openai/codex) | Yes | Run `codex login` first. Default model: `gpt-5.5` |
| Java 11+ | Yes | Runs `opendataloader-pdf` (installed with the Python dependencies) |
| Poppler (`pdftoppm`) | Yes | Renders PDF pages. Ubuntu: `apt install poppler-utils` · macOS: `brew install poppler` · Windows: `conda install -c conda-forge poppler` |

> [!WARNING]
> Codex is invoked with `--dangerously-bypass-approvals-and-sandbox` ([vision.py](plotlogic_mvp/app/services/vision.py)). Run only in a trusted local environment.

## Quick Installation

**Linux / macOS**

```bash
git clone https://github.com/BonusP/SCALE.git
cd SCALE
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/BonusP/SCALE.git
cd SCALE
py -3.12 -m venv .venv312
.venv312\Scripts\Activate.ps1
pip install -r requirements.txt
```

All dependency versions are pinned in [requirements.txt](requirements.txt). On Ubuntu, if `venv` fails with an `ensurepip` error, run `sudo apt install python3.12-venv`.

## How to Use

### 1. Start the server

Run from the repository root with the virtual environment activated.

```bash
# Linux / macOS
PYTHONUTF8=1 python -m uvicorn app.main:app --app-dir plotlogic_mvp --host 127.0.0.1 --port 8092
```

```powershell
# Windows
$env:PYTHONUTF8 = "1"
python -m uvicorn app.main:app --app-dir plotlogic_mvp --host 127.0.0.1 --port 8092
```

Then open http://127.0.0.1:8092/.

> [!IMPORTANT]
> Start the server with the virtual environment activated. Otherwise `opendataloader-pdf` is not found on `PATH`, and the server silently falls back to whole-page input (no figure detection). Alternatively, set `CMSL_OPENDATALOADER_PDF_CMD` to its full path.

### 2. Extract data

**GUI**

| Page | Purpose |
| --- | --- |
| `/` | Upload PDF, monitor jobs |
| `/papers/{paper_id}/results` | Per-paper results |
| `/debug/{job_id}/{chart_id}` | Intermediate artifacts (crops, masks, overlays) |


### 3. Outputs

```
data/
├── mvp.sqlite3                        # jobs, charts, events
├── papers/{paper_id}/source.pdf
└── jobs/{job_id}/
    ├── candidates/*.json              # VLM classification (with prompt_sha256)
    └── plotlogic_runs/{panel_id}/
        ├── curve_series_data.csv      # reconstructed data
        ├── overlay.png                # traced curves over the original
        ├── axis_mapping.json
        ├── validation_report.json
        └── run_manifest.json
```

The `data/` directory is created on first run and is git-ignored.

## ChartQA Evaluation

This evaluation uses 21 multi-series line charts from the [ChartQA](https://github.com/vis-nlp/ChartQA) **test** split ([ids.txt](experiments/chartqa/ids.txt)). Run all commands from the repository root with the virtual environment activated.

**1. Prepare data**: download ChartQA (~875 MB) and build the input PDF.

```bash
curl -L -o "ChartQA Dataset.zip" \
  "https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/af8b6f5c08c95085271561c2a3f9d15f2b5a9031/ChartQA%20Dataset.zip"
python experiments/chartqa/prepare_chartqa.py --chartqa "ChartQA Dataset.zip"
```

This writes `data/chartqa/png/`, `data/chartqa/tables/` (ground truth), and `data/chartqa/selected_images.pdf` (one chart per page).

**2. Extract**: start the server and upload the PDF.

```bash
curl -F "file=@data/chartqa/selected_images.pdf" "http://127.0.0.1:8092/api/jobs"
```

**3. Evaluate**: once the job is `completed`:

```bash
JOB=<job_id>
python experiments/chartqa/prepare_eval.py --job $JOB
python experiments/chartqa/evaluate.py --run chartqa21_$JOB --data experiments/chartqa/runs/chartqa21_$JOB/dataset
```

The metric is **RMS-F1** (Relative Mapping Similarity, DePlot; Liu et al., 2022) with τ = θ = 1.0. The mean is written to `experiments/chartqa/runs/chartqa21_<job_id>/summary.json` and per-chart scores to `per_image.csv`.

## Configuration

Environment variables, read in [config.py](plotlogic_mvp/app/config.py):

| Variable | Default | Description |
| --- | --- | --- |
| `CMSL_PORT` | `8092` | Server port |
| `PLOTLOGIC_CODEX_BIN` | `codex` on `PATH` | Codex CLI path |
| `PLOTLOGIC_MVP_CODEX_MODEL` | `gpt-5.5` | VLM model |
| `PLOTLOGIC_MVP_CODEX_REASONING_EFFORT` | `medium` | Reasoning effort |
| `PLOTLOGIC_MVP_CODEX_TIMEOUT_SECONDS` | `180` | Timeout per VLM call |
| `PLOTLOGIC_MVP_USE_CODEX` | `1` | `0` disables VLM calls (debugging) |
| `PLOTLOGIC_MVP_OCR` | `1` | Enable RapidOCR |
| `CMSL_OPENDATALOADER_PDF_CMD` | `opendataloader-pdf` | External PDF layout parser |

## Project Structure

```
.
├── plotlogic_mvp/              # FastAPI server and web UI
│   ├── app/main.py             # API routes
│   ├── app/config.py           # paths and environment variables
│   ├── app/services/           # pipeline, VLM calls, OCR, PDF input
│   ├── app/static/             # web UI
│   └── prompts/                # VLM prompts and chart-family registry
├── plotlogic/
│   └── core/                   # pixel-level curve extraction (CMSL)
├── experiments/chartqa/        # ChartQA data preparation and evaluation
└── requirements.txt            # pinned dependencies
```

## Reproducibility Notes

- **VLM outputs are non-deterministic.** Fix the model, reasoning effort, and prompt to compare runs. Each run records `prompt_sha256` in `candidates/*.json`.
- **Classification cache.** Re-running a paper reuses earlier classifications that have the same prompt hash. Delete the job (`DELETE /api/jobs/{job_id}`) to force reclassification.

## License

[GPL-3.0](LICENSE). ChartQA is distributed under GPL-3.0 by its authors.
