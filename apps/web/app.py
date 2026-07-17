# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st

from media import image_preview as load_image_preview


PROJECT_ROOT = Path(r"C:\FINAM\Conference")
BACKEND_URL = "http://127.0.0.1:8765"
DEMO_REPORT_ID = "384092"


def api_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def report_api_path(report_id: str) -> str:
    return f"{BACKEND_URL}/api/reports/{quote(report_id, safe='')}"

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


@st.cache_data(ttl=10)
def discover_reports() -> dict[str, str]:
    response = api_session().get(f"{BACKEND_URL}/api/reports", timeout=10)
    response.raise_for_status()
    reports = {}
    for item in response.json().get("reports", []):
        report_id = str(item["report_id"])
        source_name = Path(str(item.get("source_root") or report_id)).name
        status = str(item.get("worker_status") or "pending")
        reports[report_id] = f"{report_id} · {source_name} · {status}"
    if DEMO_REPORT_ID in reports:
        reports[DEMO_REPORT_ID] = "384092 · Хампинская площадь · 1979–1980"
    return dict(sorted(reports.items(), key=lambda item: (item[0] != DEMO_REPORT_ID, item[0])))


@st.cache_data(ttl=10)
def load_report_view(report_id: str) -> dict:
    session = api_session()
    status_response = session.get(f"{report_api_path(report_id)}/status", timeout=10)
    status_response.raise_for_status()
    status = status_response.json()
    source_root = Path(str(status.get("source_root") or report_id))
    result_response = session.get(report_api_path(report_id), timeout=15)
    result = result_response.json() if result_response.ok else {}
    raw_entities = result.get("entities", {})
    groups = {
        "structures": list(raw_entities.get("structures", [])) if isinstance(raw_entities, dict) else [],
        "wells": list(raw_entities.get("wells", [])) if isinstance(raw_entities, dict) else [],
        "horizons": list(raw_entities.get("horizons", [])) if isinstance(raw_entities, dict) else [],
        "survey_tasks": list(raw_entities.get("key_results", [])) if isinstance(raw_entities, dict) else [],
    }
    if isinstance(raw_entities, list):
        plural = {"structure": "structures", "well": "wells", "horizon": "horizons", "finding": "survey_tasks"}
        for entity in raw_entities:
            group = plural.get(entity.get("entity_type"))
            if group:
                groups[group].append({"name": entity.get("name", ""), **entity.get("attributes", {})})
    map_response = session.get(f"{report_api_path(report_id)}/maps", timeout=10)
    maps = map_response.json() if map_response.ok else {"sources": [], "digitized": [], "metrics": {}}
    return {
        "report": {
            "id": report_id,
            "title": source_root.name,
            "survey_party": "Архивный геолого-геофизический отчёт",
            "years": "не определено",
            "region": str(source_root.parent.name),
            "status": result.get("status", status.get("worker", {}).get("status", "pending")),
        },
        "summary": {
            "structures": len(groups["structures"]),
            "wells": len(groups["wells"]),
            "horizons": len(groups["horizons"]),
        },
        "entities": groups,
        "maps": maps,
    }


@st.cache_data(ttl=5)
def load_processing_state(report_id: str) -> dict:
    response = api_session().get(f"{report_api_path(report_id)}/status", timeout=10)
    if not response.ok:
        return {"worker": {}, "privacy": {}, "manifest": {}}
    status = response.json()
    return {
        "worker": status.get("worker", {}),
        "privacy": {},
        "manifest": {"summary": status.get("summary", {}), "source_root": status.get("source_root")},
    }


@st.cache_data
def _cached_image_preview(
    path: str, modified_ns: int, max_size: tuple[int, int] = (1800, 1100)
):
    return load_image_preview(path, max_size)


def resolve_artifact_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def image_preview(path: str, max_size: tuple[int, int] = (1800, 1100)):
    source = resolve_artifact_path(path)
    modified_ns = source.stat().st_mtime_ns if source.exists() else 0
    return _cached_image_preview(str(source), modified_ns, max_size)


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


def submit_archive(upload) -> dict:
    upload.seek(0)
    response = api_session().post(
        f"{BACKEND_URL}/api/reports/upload",
        files={"file": (clean_filename(upload.name), upload, upload.type or "application/octet-stream")},
        timeout=60 * 60,
    )
    response.raise_for_status()
    return response.json()


def run_report(report_id: str) -> dict:
    response = api_session().post(
        f"{report_api_path(report_id)}/run",
        json={"force": []},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


@st.fragment(run_every="7s")
def render_processing_status(report_id: str) -> None:
    processing = load_processing_state(report_id)
    worker_status = processing["worker"].get("status")
    page_count = processing["manifest"].get("summary", {}).get("page_count")
    if worker_status in {None, "pending"}:
        st.info(f"В очереди на обработку · {page_count or 0} стр.")
    elif worker_status == "blocked":
        st.warning(f"Зарегистрирован · {page_count or 0} стр. · ожидает повторного запуска")
    elif worker_status == "running":
        st.info(f"Обрабатывается · {page_count or 0} стр.")
    elif worker_status == "failed":
        failed_stages = [
            (name, stage)
            for name, stage in processing["worker"].get("stages", {}).items()
            if stage.get("status") == "failed"
        ]
        if failed_stages:
            stage_name, failed = failed_stages[0]
            detail = failed.get("detail") or failed.get("error") or "неизвестная ошибка"
            st.error(f"Сбой на стадии {stage_name}: {detail}")
        else:
            st.error("Обработка остановлена")
    elif worker_status == "completed" and report_id != DEMO_REPORT_ID:
        st.success(f"Основная обработка завершена · {page_count or 0} стр.")
    if report_id != DEMO_REPORT_ID and worker_status in {"blocked", "failed"}:
        if st.button("Повторить с места сбоя", key=f"run:{report_id}", use_container_width=True):
            try:
                run_report(report_id)
                load_processing_state.clear()
                discover_reports.clear()
                st.rerun(scope="fragment")
            except requests.RequestException as error:
                st.error(f"Не удалось повторить: {error}")


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


def ask_corpus(question: str) -> dict:
    response = api_session().post(
        f"{BACKEND_URL}/api/search",
        json={"question": question, "mode": "live"},
        timeout=120,
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
    render_processing_status(report_id)

with upload_column:
    st.markdown('<div class="ga-section">Новый архив</div>', unsafe_allow_html=True)
    upload = st.file_uploader(
        "Перетащите архив или комплект документов",
        type=["zip", "pdf", "jpg", "jpeg", "png", "tif", "tiff"],
        help="Файл сначала сохраняется в локальную очередь.",
    )
    if upload is not None:
        payload = upload.getvalue() if upload.size <= 50 * 1024 * 1024 else b""
        classification = classify_upload(upload.name, payload) if payload else {"Архив": 1}
        active = {name: count for name, count in classification.items() if count}
        st.caption("Предварительная локальная классификация")
        cols = st.columns(max(1, len(active)))
        for column, (name, count) in zip(cols, active.items()):
            column.metric(name, count)
        if st.button("Добавить в локальную очередь", type="primary", use_container_width=True):
            try:
                result = submit_archive(upload)
                discover_reports.clear()
                load_processing_state.clear()
                st.success(f"Загружено: {result.get('uploaded_name')}. Зарегистрировано отчётов: {result['registered']}")
                st.rerun()
            except requests.RequestException as error:
                st.error(f"Ошибка загрузки: {error}")

st.markdown('<div class="ga-section">Обработка отчёта</div>', unsafe_allow_html=True)
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


map_state = bundle.get("maps", {}).get("status")
agents = [
    ("01", "OCR → Markdown", "PaddleOCR читает текстовые страницы и сохраняет единый Markdown.", worker_stage_state("markdown") or "В ОЧЕРЕДИ"),
    ("02", "Поиск по отчёту", "Markdown индексируется в RAGFlow и становится доступен для вопросов.", worker_stage_state("ragflow") or "В ОЧЕРЕДИ"),
    ("03", "Карты", "Исходные карты выделяются отдельно; готовая векторизация показывается рядом.", "ГОТОВО" if map_state == "completed" else "НЕ ОЦИФРОВАНО"),
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

overview_tab, materials_tab, search_tab, corpus_tab = st.tabs(
    ["Результаты", "Карты", "Поиск по отчёту", "Поиск по фонду"]
)

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
    maps = bundle.get("maps", {})
    source_maps = [item for item in maps.get("sources", []) if item.get("exists")]
    digitized_maps = [item for item in maps.get("digitized", []) if item.get("exists")]
    page_ids = list(dict.fromkeys(item.get("page_id") for item in source_maps if item.get("page_id")))
    if not page_ids:
        st.info("Карты в отчёте не обнаружены.")
    else:
        default_page = "page:00184" if "page:00184" in page_ids else page_ids[0]
        selected_page = st.selectbox(
            "Лист",
            page_ids,
            index=page_ids.index(default_page),
            key="map-page-v2",
            format_func=lambda page_id: next(
                (item.get("label") or page_id for item in source_maps if item.get("page_id") == page_id),
                page_id,
            ),
        )
        source = next((item for item in source_maps if item.get("page_id") == selected_page), None)
        page_artifacts = [item for item in digitized_maps if item.get("page_id") == selected_page]
        result = next(
            (item for item in page_artifacts if item.get("name") == "surface_clean_preview"),
            None,
        )
        source_column, result_column = st.columns(2, gap="large")
        with source_column:
            st.subheader("Исходная карта")
            source_preview = image_preview(source["path"]) if source else None
            if source_preview is not None:
                st.image(source_preview, width="stretch")
        with result_column:
            st.subheader("Оцифрованная карта")
            result_preview = image_preview(result["path"]) if result else None
            if result_preview is not None:
                st.image(result_preview, width="stretch")
            else:
                st.info("Этот лист пока не оцифрован.")

        downloads = [
            item
            for item in page_artifacts
            if item.get("name") in {"interpolated_contours", "local_cps3"}
        ]
        download_columns = st.columns(max(1, len(downloads)))
        for index, (column, artifact) in enumerate(zip(download_columns, downloads)):
            path = resolve_artifact_path(str(artifact.get("path") or ""))
            if path.is_file():
                column.download_button(
                    "Скачать GeoJSON" if artifact.get("name") == "interpolated_contours" else "Скачать CPS-3",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=str(artifact.get("media_type") or "application/octet-stream"),
                    key=f"map-download:{report_id}:{selected_page}:{index}",
                    use_container_width=True,
                )

with search_tab:
    st.subheader("Вопрос к отчёту")
    search_available = worker_stages.get("ragflow", {}).get("status") == "completed"
    if not search_available:
        st.info("Поиск станет доступен после индексации Markdown в RAGFlow.")
    preset_columns = st.columns(3)
    presets = [
        "Какие структуры выделены в отчёте?",
        "Какие скважины упоминаются и что в них установлено?",
        "Какие отражающие горизонты изучались?",
    ]
    question_key = f"question:{report_id}"
    for column, preset in zip(preset_columns, presets):
        if column.button(preset, key=f"preset:{report_id}:{preset}", use_container_width=True, disabled=not search_available):
            st.session_state[question_key] = preset
    question = st.text_input(
        "Вопрос",
        key=question_key,
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
            citations = answer.get("evidence") or answer.get("citations") or answer.get("sources") or []
            if citations:
                with st.expander("Источники"):
                    for citation in citations:
                        pages = ", ".join(citation.get("evidence", []))
                        st.caption(f"{pages} · релевантность {citation.get('similarity', 0):.2f}")
                        st.write(citation.get("excerpt", ""))
        except requests.RequestException as error:
            st.error(f"Поиск не выполнился: {error}")

with corpus_tab:
    st.subheader("Поиск по всем отчётам")
    corpus_key = "corpus-question"
    preset = "В каких отчётах и скважинах были получены притоки нефти, газа или воды?"
    if st.button(preset, key="corpus-preset", use_container_width=True):
        st.session_state[corpus_key] = preset
    question = st.text_input(
        "Вопрос по фонду",
        key=corpus_key,
        placeholder="Например: где были получены притоки нефти?",
        label_visibility="collapsed",
    )
    if st.button("Найти по всем отчётам", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("RAGFlow ищет по фонду..."):
                answer = ask_corpus(question.strip())
            st.markdown(answer.get("answer") or "Ответ не найден.")
            if answer.get("evidence"):
                with st.expander("Источники"):
                    for source in answer["evidence"]:
                        pages = ", ".join(source.get("evidence", []))
                        st.caption(
                            f"Отчёт {source.get('report_id') or 'не определён'} · {pages} · "
                            f"релевантность {source.get('similarity', 0):.2f}"
                        )
                        st.write(source.get("excerpt", ""))
        except requests.RequestException as error:
            st.error(f"Поиск по фонду не выполнился: {error}")
