from __future__ import annotations

import os
# ###수정 Windows에서도 실행 가능한 Codex CLI 경로를 탐색합니다.
import shutil
# ###
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# ###수정 실험 경로의 jobs·papers·SQLite를 원본 데이터와 분리합니다.
EXPERIMENT_ROOT = PROJECT_ROOT.parent
MONOREPO_ROOT = EXPERIMENT_ROOT.parents[1]
DATA_ROOT = EXPERIMENT_ROOT / "data"
DB_PATH = DATA_ROOT / "mvp.sqlite3"
PAPERRAG_ROOT = MONOREPO_ROOT / "PaperRAG"
PAPERRAG_PAPERS_DIR = DATA_ROOT / "papers"
PAPERRAG_MANIFEST_PATH = DATA_ROOT / "manifests" / "papers.jsonl"
PAPERRAG_OPENDATALOADER_CMD = os.getenv("CMSL_OPENDATALOADER_PDF_CMD", "opendataloader-pdf")
PAPERRAG_JAVA_HOME = os.getenv("JAVA_HOME", "")
PLOTLOGIC_ROOT = EXPERIMENT_ROOT / "plotlogic"
PROMPTS_ROOT = PROJECT_ROOT / "prompts"
CMSL_PORT = int(os.getenv("CMSL_PORT", "8092"))
PDF_IMAGE_FALLBACK_ENABLED = os.getenv("CMSL_PDF_IMAGE_FALLBACK", "1").lower() not in {"0", "false", "no"}
# ###

# ###수정 실험의 VLM 호출에 실행 가능한 Codex CLI 파일을 사용합니다.
CODEX_BIN = os.getenv("PLOTLOGIC_CODEX_BIN", shutil.which("codex") or "codex")
# ###
CODEX_MODEL = os.getenv("PLOTLOGIC_MVP_CODEX_MODEL", "gpt-5.5")
CODEX_REASONING_EFFORT = os.getenv("PLOTLOGIC_MVP_CODEX_REASONING_EFFORT", "medium")
# ###수정 로컬 CLI가 거부하는 global service_tier=default를 실험 호출에서만 override합니다.
CODEX_SERVICE_TIER = os.getenv("CMSL_CODEX_SERVICE_TIER", "fast")
# ###
CODEX_TIMEOUT_SECONDS = int(os.getenv("PLOTLOGIC_MVP_CODEX_TIMEOUT_SECONDS", "180"))
USE_CODEX = os.getenv("PLOTLOGIC_MVP_USE_CODEX", "1").lower() not in {"0", "false", "no"}

# ###수정 실험의 기본 classifier를 CMSL prompt로 지정합니다.
# The baseline prompt is the comparison point for every prompt experiment, so it is
# never edited in place. Point this at another template to run a variant instead.
VISION_PROMPT_TEMPLATE = os.getenv(
    "PLOTLOGIC_MVP_VISION_PROMPT", "figure_panel_classifier_cmsl.md"
)

OCR_ENABLED = os.getenv("PLOTLOGIC_MVP_OCR", "1").lower() not in {"0", "false", "no"}
OCR_BACKEND = os.getenv("PLOTLOGIC_MVP_OCR_BACKEND", "rapidocr")
OCR_MIN_CONFIDENCE = float(os.getenv("PLOTLOGIC_MVP_OCR_MIN_CONFIDENCE", "0.5"))

# ###수정 원본 PaperRAG 저장소에 쓰는 semantic·index 후처리는 실행하지 않습니다.
PAPERRAG_POSTPROCESS_ENABLED = False
# ###
PAPERRAG_POSTPROCESS_TIMEOUT_SECONDS = int(os.getenv("PLOTLOGIC_MVP_PAPERRAG_POSTPROCESS_TIMEOUT_SECONDS", "1800"))
PAPERRAG_SEMANTIC_BACKEND = os.getenv("PLOTLOGIC_MVP_PAPERRAG_SEMANTIC_BACKEND", "codex-exec")
PAPERRAG_SEMANTIC_MODEL = os.getenv("PLOTLOGIC_MVP_PAPERRAG_SEMANTIC_MODEL", "gpt-5.5")
PAPERRAG_KOREAN_SUMMARY_MODEL = os.getenv("PLOTLOGIC_MVP_PAPERRAG_KOREAN_SUMMARY_MODEL", PAPERRAG_SEMANTIC_MODEL)


def ensure_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "jobs").mkdir(parents=True, exist_ok=True)
    PAPERRAG_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    PAPERRAG_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
