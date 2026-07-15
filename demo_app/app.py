# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image


Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(r"C:\FINAM\Conference")
BUNDLE_PATH = Path(
    r"C:\Users\Finam\Documents\Codex\2026-07-13\new-chat\outputs\backend"
    r"\report_384092_bundle.json"
)
INBOX = PROJECT_ROOT / "reports_inbox"
RUNS_ROOT = PROJECT_ROOT / "runs"
BACKEND_URL = "http://127.0.0.1:8765"
DEMO_REPORT_ID = "384092"

st.set_page_config(page_title="GeoArchive AI", page_icon="GA", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background: #f5f6f7; color: #172027; }
      .block-container { max-width: 1180px; padding-top: 4.5rem; padding-bottom: 3rem; }
      h1, h2, h3, p, label, button { letter-spacing: 0 !important; }
      div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #dfe3e6; border-radius: 6px; padding: 14px 16px;
      }
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff; border-color: #dfe3e6; border-radius: 6px;
      }
      div[data-testid="stFileUploaderDropzone"] {
        background: #ffffff; border: 1px dashed #89949c; border-radius: 6px;
      }
      button[kind="primary"] { background: #0f6b5d; border-radius: 5px; }
      button[kind="secondary"] { border-radius: 5px; }
      .ga-brand { display:flex; align-items:center; gap:14px; margin-bottom: 8px; }
      .ga-mark {
        width:42px; height:42px; display:flex; align-items:center; justify-content:center;
        background:#123f3a; color:white; font-weight:700; border-radius:6px;
      }
      .ga-title { font-size:28px; font-weight:700; line-height:1.1; }
      .ga-subtitle { color:#5d6870; font-size:14px; margin-top:3px; }
      .ga-section { font-size:18px; font-weight:650; margin: 22px 0 10px; }
      .agent-index {
        display:inline-flex; width:24px; height:24px; align-items:center; justify-content:center;
        border-radius:50%; background:#d9ebe7; color:#174f47; font-size:12px; font-weight:700;
      }
      .agent-name { font-weight:650; margin:10px 0 5px; }
      .agent-copy { color:#657078; font-size:13px; line-height:1.45; min-height:58px; }
      .agent-state { margin-top:10px; font-size:12px; font-weight:650; color:#0f6b5d; }
      .report-kicker { color:#0f6b5d; font-size:12px; font-weight:700; text-transform:uppercase; }
      .report-title { font-size:23px; font-weight:700; margin:5px 0; }
      .report-meta { color:#657078; font-size:14px; }
      #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_demo_bundle() -> dict:
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))


@st.cache_data(ttl=10)
def discover_reports() -> dict[str, str]:
    reports: dict[str, str] = {}
    for manifest_path in sorted(RUNS_ROOT.glob("*/job.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        report_id = str(manifest.get("report_id") or manifest_path.parent.name)
        source_name = Path(str(manifest.get("source_root") or report_id)).name
        reports[report_id] = f"{report_id} · {source_name}"
    if DEMO_REPORT_ID in reports:
        reports[DEMO_REPORT_ID] = "384092 · Хампинская площадь · 1979–1980"
    return dict(sorted(reports.items(), key=lambda item: (item[0] != DEMO_REPORT_ID, item[0])))


@st.cache_data(ttl=10)
def load_report_view(report_id: str) -> dict:
    if report_id == DEMO_REPORT_ID:
        bundle = dict(load_demo_bundle())
        bundle["artifacts"] = list(bundle.get("artifacts", []))
        map_result_path = RUNS_ROOT / report_id / "map_agent" / "result.json"
        if map_result_path.exists():
            try:
                map_result = json.loads(map_result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                map_result = {}
            bundle["map_qc"] = map_result.get("metrics", {})
            for artifact in map_result.get("artifacts", []):
                media_type = str(artifact.get("media_type") or "")
                bundle["artifacts"].append(
                    {
                        "id": f"map-agent:{artifact.get('name', 'artifact')}",
                        "label": artifact.get("label") or artifact.get("name") or "Результат картографа",
                        "type": "image" if media_type.startswith("image/") else "vector",
                        "path": artifact.get("path"),
                    }
                )
        return bundle
    run_dir = RUNS_ROOT / report_id
    manifest_path = run_dir / "job.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_path = run_dir / "orchestrator" / "report_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    groups = {"structures": [], "wells": [], "horizons": [], "survey_tasks": []}
    plural = {"structure": "structures", "well": "wells", "horizon": "horizons", "finding": "survey_tasks"}
    for entity in result.get("entities", []):
        group = plural.get(entity.get("entity_type"))
        if not group:
            continue
        attributes = dict(entity.get("attributes", {}))
        item = {"name": entity.get("name", ""), **attributes, "evidence": entity.get("evidence", [])}
        groups[group].append(item)
    source_root = Path(manifest["source_root"])
    contact_sheet = run_dir / "graphics_contact_sheet.jpg"
    artifacts = []
    if contact_sheet.exists():
        artifacts.append(
            {
                "id": f"{report_id}:contact-sheet",
                "label": "Обзор графических материалов",
                "type": "source_scan",
                "path": str(contact_sheet),
            }
        )
    return {
        "report": {
            "id": report_id,
            "title": source_root.name,
            "survey_party": "Архивный геолого-геофизический отчёт",
            "years": "не определено",
            "region": str(source_root.parent.name),
            "status": result.get("status", "processing"),
        },
        "summary": {
            "structures": len(groups["structures"]),
            "wells": len(groups["wells"]),
            "horizons": len(groups["horizons"]),
        },
        "entities": groups,
        "artifacts": artifacts,
        "map_qc": result.get("map", {}).get("metrics", {}),
    }


def load_operator_state(report_id: str) -> dict:
    path = RUNS_ROOT / report_id / "orchestrator" / "state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(ttl=5)
def load_processing_state(report_id: str) -> dict:
    run_dir = RUNS_ROOT / report_id
    payload = {"worker": {}, "privacy": {}, "manifest": {}}
    for key, path in (
        ("worker", run_dir / "worker" / "state.json"),
        ("privacy", run_dir / "privacy.json"),
        ("manifest", run_dir / "job.json"),
    ):
        if not path.exists():
            continue
        try:
            payload[key] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return payload


@st.cache_data
def image_preview(path: str, max_size: tuple[int, int] = (1800, 1100)) -> Image.Image | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    with Image.open(candidate) as source:
        image = source.convert("RGB")
        image.thumbnail(max_size)
        return image.copy()


def clean_filename(name: str) -> str:
    filename = Path(name).name
    return re.sub(r"[^0-9A-Za-zА-Яа-яЁё._() -]+", "_", filename).strip(" .") or "archive.zip"


def classify_upload(name: str, payload: bytes) -> dict[str, int]:
    categories = {"Текстовые документы": 0, "Карты и графика": 0, "Таблицы": 0, "Прочее": 0}
    filenames = [name]
    if name.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                filenames = [item.filename for item in archive.infolist() if not item.is_dir()]
        except zipfile.BadZipFile:
            return {"Прочее": 1}
    for filename in filenames:
        lower = filename.lower()
        suffix = Path(lower).suffix
        is_map = any(word in lower for word in ("карт", "схем", "разрез", "профил", "map", "section"))
        if suffix in {".doc", ".docx", ".pdf", ".txt", ".rtf"} and not is_map:
            categories["Текстовые документы"] += 1
        elif suffix in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".svg"} or is_map:
            categories["Карты и графика"] += 1
        elif suffix in {".xls", ".xlsx", ".csv", ".dbf"}:
            categories["Таблицы"] += 1
        else:
            categories["Прочее"] += 1
    return categories


def submit_archive(upload) -> Path:
    INBOX.mkdir(parents=True, exist_ok=True)
    filename = clean_filename(upload.name)
    target = INBOX / filename
    if target.exists():
        target = INBOX / f"{datetime.now():%Y%m%d_%H%M%S}_{filename}"
    target.write_bytes(upload.getvalue())
    return target


def ask_report(report_id: str, question: str) -> dict:
    session = requests.Session()
    session.trust_env = False
    response = session.post(
        f"{BACKEND_URL}/api/reports/{report_id}/ask",
        json={"question": question, "mode": "live"},
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


st.markdown(
    """
    <div class="ga-brand">
      <div class="ga-mark">GA</div>
      <div>
        <div class="ga-title">GeoArchive AI</div>
        <div class="ga-subtitle">Рабочее пространство для архивных геолого-геофизических данных</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

source_column, upload_column = st.columns([0.9, 1.45], gap="large")
with source_column:
    st.markdown('<div class="ga-section">Архивный фонд</div>', unsafe_allow_html=True)
    available_reports = discover_reports()
    report_id = st.selectbox(
        "Выбранный пример",
        options=list(available_reports),
        format_func=lambda value: available_reports[value],
        label_visibility="collapsed",
    )
    bundle = load_report_view(report_id)
    report = bundle["report"]
    summary = bundle["summary"]
    st.markdown(
        f"""
        <div class="report-kicker">Обработанный пример</div>
        <div class="report-title">{report['title']}</div>
        <div class="report-meta">{report['survey_party']} · {report['region']}</div>
        """,
        unsafe_allow_html=True,
    )
    processing = load_processing_state(report_id)
    worker_status = processing["worker"].get("status")
    page_count = processing["manifest"].get("summary", {}).get("page_count")
    if worker_status == "blocked":
        st.warning(f"Зарегистрирован · {page_count or 0} стр. · ожидает повторного запуска")
    elif worker_status == "running":
        st.info(f"Обрабатывается · {page_count or 0} стр.")
    elif worker_status == "failed":
        st.error("Обработка остановлена на одной из стадий")
    elif worker_status == "completed" and report_id != DEMO_REPORT_ID:
        st.success(f"Основная обработка завершена · {page_count or 0} стр.")

with upload_column:
    st.markdown('<div class="ga-section">Новый архив</div>', unsafe_allow_html=True)
    upload = st.file_uploader(
        "Перетащите архив или комплект документов",
        type=["zip", "pdf", "jpg", "jpeg", "png", "tif", "tiff"],
        help="Файл сначала сохраняется в локальную очередь.",
    )
    if upload is not None:
        payload = upload.getvalue()
        classification = classify_upload(upload.name, payload)
        active = {name: count for name, count in classification.items() if count}
        st.caption("Предварительная локальная классификация")
        cols = st.columns(max(1, len(active)))
        for column, (name, count) in zip(cols, active.items()):
            column.metric(name, count)
        if st.button("Добавить в локальную очередь", type="primary", use_container_width=True):
            saved = submit_archive(upload)
            st.success(f"Архив добавлен: {saved.name}")

st.markdown('<div class="ga-section">Конвейер обработки</div>', unsafe_allow_html=True)
operator_state = load_operator_state(report_id)
operator_agents = operator_state.get("agents", {})
processing_state = load_processing_state(report_id)
worker_stages = processing_state.get("worker", {}).get("stages", {})


def worker_stage_state(name: str) -> str | None:
    status = worker_stages.get(name, {}).get("status")
    if status == "completed":
        return "ГОТОВ"
    if status == "blocked":
        return "ОЖИДАЕТ РАЗРЕШЕНИЯ"
    if status == "running":
        return "В РАБОТЕ"
    if status == "failed":
        return "ТРЕБУЕТ ВНИМАНИЯ"
    if status == "pending":
        return "В ОЧЕРЕДИ"
    return None


def agent_state(name: str, fallback: str) -> str:
    record = operator_agents.get(name, {})
    status = record.get("status")
    if status == "completed":
        result = record.get("result", {})
        metrics = result.get("metrics", {})
        if any(issue.get("severity") == "review" for issue in result.get("issues", [])):
            return "ТРЕБУЕТ ПРОВЕРКИ"
        if metrics.get("quality_status") == "review":
            return "ТРЕБУЕТ ПРОВЕРКИ"
        result_status = metrics.get("status")
        return "ТРЕБУЕТ ПРОВЕРКИ" if result_status in {"partial", "review"} else "ГОТОВ"
    if status == "blocked":
        return "ОЖИДАЕТ ИНСТРУМЕНТ"
    if status == "running":
        return "В РАБОТЕ"
    if status == "failed":
        return "ТРЕБУЕТ ВНИМАНИЯ"
    if status == "pending":
        return "В ОЧЕРЕДИ"
    return fallback


agents = [
    ("01", "Классификатор", "Разделяет тома, текст, карты, разрезы и таблицы.", worker_stage_state("page_routing") or agent_state("classifier", "ГОТОВ")),
    ("02", "RAG-агент", "Извлекает структуры, скважины, горизонты и выводы с источниками.", worker_stage_state("agents") or agent_state("rag", "ГОТОВ")),
    ("03", "Картограф", "Передаёт карты инструментам сегментации, векторизации и привязки.", agent_state("map", "ПОДКЛЮЧЕНИЕ OSS")),
    ("04", "QC-агент", "Сверяет одинаковые сущности между текстом, картами и таблицами.", agent_state("qc", "ПРОТОТИП")),
    ("05", "Экспорт", "Собирает проверенные данные в GeoPackage, GeoJSON и отчёт.", agent_state("export", worker_stage_state("bundle") or "СЛЕДУЮЩИЙ ЭТАП")),
]
agent_columns = st.columns(len(agents), gap="small")
for column, (index, name, copy, state) in zip(agent_columns, agents):
    with column:
        with st.container(border=True):
            st.markdown(
                f"""
                <span class="agent-index">{index}</span>
                <div class="agent-name">{name}</div>
                <div class="agent-copy">{copy}</div>
                <div class="agent-state">{state}</div>
                """,
                unsafe_allow_html=True,
            )

overview_tab, materials_tab, search_tab = st.tabs(["Результаты", "Материалы", "Поиск по отчёту"])

with overview_tab:
    metric_columns = st.columns(4)
    metric_columns[0].metric("Структуры", summary["structures"])
    metric_columns[1].metric("Скважины", summary["wells"])
    metric_columns[2].metric("Горизонты", summary["horizons"])
    metric_columns[3].metric("Полевые работы", report["years"])

    structures_column, wells_column = st.columns(2, gap="large")
    with structures_column:
        st.subheader("Структуры")
        structure_rows = [
            {
                "Название": item.get("name", ""),
                "Статус": item.get("status", ""),
                "Горизонты": ", ".join(item.get("horizons", [])),
            }
            for item in bundle["entities"]["structures"]
        ]
        st.dataframe(pd.DataFrame(structure_rows), hide_index=True, width="stretch")
    with wells_column:
        st.subheader("Скважины")
        well_rows = [
            {
                "Скважина": item.get("name", ""),
                "Площадь": item.get("area", ""),
                "Результат испытаний": item.get("test_result", ""),
            }
            for item in bundle["entities"]["wells"]
        ]
        st.dataframe(pd.DataFrame(well_rows), hide_index=True, width="stretch")

with materials_tab:
    st.subheader("Материалы отчёта")
    source_artifacts = [item for item in bundle["artifacts"] if item.get("type") == "source_scan"]
    if source_artifacts:
        preview = image_preview(source_artifacts[0]["path"])
        if preview is not None:
            st.image(preview, caption=source_artifacts[0]["label"], width="stretch")
    map_artifacts = [item for item in bundle["artifacts"] if item.get("id", "").startswith("map-agent:")]
    image_artifacts = [item for item in map_artifacts if item.get("type") == "image"]
    for artifact in image_artifacts:
        preview = image_preview(artifact["path"])
        if preview is not None:
            st.image(preview, caption=artifact["label"], width="stretch")
    if map_artifacts:
        map_qc = bundle.get("map_qc", {})
        if map_qc:
            qc_columns = st.columns(4)
            qc_columns[0].metric("Изогипсы", map_qc.get("isoline_features", 0))
            qc_columns[1].metric(
                "Со значением",
                f"{map_qc.get('valued_total', 0)} / {map_qc.get('isoline_features', 0)}",
            )
            qc_columns[2].metric("Пересечения", map_qc.get("crossing_isoline_pairs", 0))
            qc_columns[3].metric("Конфликты", map_qc.get("crosscheck_disagreements", 0))
        st.warning(
            "Картографический слой экспериментальный: координаты пока пиксельные, "
            "геопривязка и замкнутые структуры не подтверждены."
        )
        for artifact in map_artifacts:
            path = Path(str(artifact.get("path") or ""))
            if artifact.get("type") == "vector" and path.exists():
                st.download_button(
                    artifact["label"],
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/geo+json" if path.suffix.casefold() == ".geojson" else "application/json",
                    key=f"download-{artifact['id']}",
                )
    else:
        st.info("Для выбранного отчёта картографический агент ещё не сформировал проверяемый слой.")

with search_tab:
    st.subheader("Вопрос к отчёту")
    search_available = report_id == DEMO_REPORT_ID
    if not search_available:
        st.info("Поиск станет доступен после завершения RAG-агента для этого фонда.")
    preset_columns = st.columns(3)
    presets = [
        "Какие структуры выделены в отчёте?",
        "Какие скважины упоминаются и что в них установлено?",
        "Какие отражающие горизонты изучались?",
    ]
    for column, preset in zip(preset_columns, presets):
        if column.button(preset, use_container_width=True, disabled=not search_available):
            st.session_state["question"] = preset
    question = st.text_input(
        "Вопрос",
        key="question",
        placeholder="Например: какие структуры рекомендуются для дальнейшего изучения?",
        label_visibility="collapsed",
    )
    if st.button(
        "Найти в отчёте",
        type="primary",
        disabled=not search_available or not question.strip(),
    ):
        try:
            with st.spinner("Проверяем материалы отчёта..."):
                answer = ask_report(report_id, question.strip())
            st.markdown(answer.get("answer") or answer.get("text") or "Ответ не найден.")
            citations = answer.get("citations") or answer.get("sources") or []
            if citations:
                with st.expander("Источники"):
                    st.json(citations)
        except requests.RequestException:
            st.error("Сервис поиска сейчас недоступен. Материалы выбранного отчёта остаются доступны локально.")
