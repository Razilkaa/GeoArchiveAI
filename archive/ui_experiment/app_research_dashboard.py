# -*- coding: utf-8 -*-
"""Demo: единый путь работы с отчётом 384092.
Скан -> OCR/RAG -> тематические агенты -> evidence -> экспериментальная карта.

Единственный источник данных фронта:
outputs/backend/report_384092_bundle.json (схема: report_bundle.schema.json).
Никаких API-вызовов, ничего не пишем."""
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

BUNDLE = Path(r"C:\Users\Finam\Documents\Codex\2026-07-13\new-chat\outputs\backend"
              r"\report_384092_bundle.json")
FACTORY_BUNDLE = Path(r"C:\FINAM\Conference\runs\384092\result_bundle.json")
BACKEND_URL = "http://127.0.0.1:8765"
BACKEND_SESSION = requests.Session()
BACKEND_SESSION.trust_env = False

st.set_page_config(page_title="Отчёт 384092 — демо", layout="wide", page_icon="🗂️")


@st.cache_data
def load_bundle():
    return json.loads(BUNDLE.read_text(encoding="utf-8"))


@st.cache_data
def load_factory_bundle(path, modified_ns):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data
def load_image(path_str, max_w=1600):
    p = Path(path_str)
    if not p.exists():
        return None
    im = Image.open(p)
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)))
    return im


def ask_backend(question, mode):
    response = BACKEND_SESSION.post(
        f"{BACKEND_URL}/api/reports/384092/ask",
        json={"question": question, "mode": mode},
        timeout=80,
    )
    response.raise_for_status()
    return response.json()


def backend_health():
    try:
        response = BACKEND_SESSION.get(f"{BACKEND_URL}/health", timeout=1.5)
        return response.ok
    except requests.RequestException:
        return False


B = load_bundle()
report = B["report"]
summary = B["summary"]
sources = {s["id"]: s for s in B["sources"]}
artifacts = B["artifacts"]
STATUS_ICONS = {"completed": "✅", "verified": "✅", "experimental": "⚠️", "partial": "🟡"}


def source_buttons(evidence_ids, key_prefix):
    """Кнопки «Показать источник» для списка evidence-id из бандла."""
    known = [sources[e] for e in evidence_ids if e in sources]
    if not known:
        st.caption("Источники: " + ", ".join(evidence_ids))
        return
    cols = st.columns(min(4, len(known)))
    for k, src in enumerate(known[:4]):
        with cols[k]:
            if st.button(f"📄 {src['label']}", key=f"{key_prefix}_{k}",
                         disabled=not src["exists"]):
                st.session_state["page_viewer"] = src["path"]
                st.session_state["page_viewer_label"] = src["label"]


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"### Отчёт **{report['id']}**")
    st.caption(f"{report['survey_party']}, {report['years']} гг. {report['region']}")
    st.divider()
    st.markdown("**Статус обработки**")
    for stage in B["pipeline"]:
        icon = STATUS_ICONS.get(stage["status"], "•")
        api = " · внешний API" if stage["calls_external_api"] else ""
        st.markdown(f"{icon} {stage['label']} — `{stage['status']}`{api}")
    st.divider()
    st.markdown("**Слои карты**")
    layer_labels = [a["label"] for a in artifacts]
    map_layer = st.radio("Слой", layer_labels, label_visibility="collapsed")
    st.divider()
    st.caption(
        "Готовые примеры хранятся локально. Свободный вопрос использует "
        "сетевой retrieval или локальный резервный поиск."
    )

# viewer источников
if st.session_state.get("page_viewer"):
    with st.expander(f"📄 {st.session_state.get('page_viewer_label', 'Источник')}",
                     expanded=True):
        img = load_image(st.session_state["page_viewer"])
        if img is not None:
            st.image(img)
        else:
            st.error("Файл не найден: " + st.session_state["page_viewer"])
        if st.button("Закрыть источник"):
            st.session_state["page_viewer"] = None
            st.rerun()

tabs = st.tabs(["Обзор", "Поиск по отчёту", "Сущности", "Карта", "Агенты",
                "Контроль качества", "Автозапуск"])

# ---------------------------------------------------------------- 1. Обзор
with tabs[0]:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader(report["title"])
        st.markdown(
            f"**{report['survey_party']}**, {report['years']} гг.\n\n"
            f"{report['region']}. Отражающие горизонты: **К** (кровля нижнего кембрия) "
            f"и **КВ** (венд-рифей, условный).\n\n"
            "Итог работ: оконтурены Тююердехская, Меикская/Меинская и Западно-Таландинская "
            "структуры, уточнена Усть-Меикско-Сыгдахская антиклинальная зона, впервые "
            "выполнены построения по горизонту КВ. Отчёт принят НТС с оценкой «отлично» "
            "(источники: том 1, стр. 3, 154)."
        )
        st.info(f"Статус обработки: `{report['status']}`")
    with c2:
        m1, m2 = st.columns(2)
        m1.metric("Структуры", summary["structures"])
        m2.metric("Скважины", summary["wells"])
        m1.metric("Горизонты", summary["horizons"])
        m2.metric("Страниц OCR", summary["text_pages_ocr"])
        m1.metric("RAG-чанков", summary["rag_chunks"])
        m2.metric("RAG-токенов", f"{summary['rag_tokens']:,}")
    st.divider()
    st.markdown("**Задачи партии (из отчёта):**")
    for t in B["entities"]["survey_tasks"]:
        st.markdown(f"- {t['text']}")

# ---------------------------------------------------------------- 2. Поиск
with tabs[1]:
    st.subheader("Поиск по отчёту")
    backend_online = backend_health()
    st.caption(
        "Свободный вопрос использует RAGFlow, а при недоступной сети автоматически "
        "переходит на локальный BM25-корпус. Готовые вопросы воспроизводятся из "
        "сохранённого проверенного прогона."
    )
    question = st.text_input(
        "Вопрос",
        placeholder="Например: какие структуры оконтурены по горизонтам К и КВ?",
    )
    mode_label = st.radio(
        "Режим",
        ["Ответ с доказательствами", "Только найти фрагменты"],
        horizontal=True,
        label_visibility="collapsed",
    )
    ask_col, status_col = st.columns([1, 3])
    with ask_col:
        run_question = st.button(
            "Найти",
            type="primary",
            use_container_width=True,
            disabled=not question.strip() or not backend_online,
        )
    with status_col:
        if backend_online:
            st.caption("Backend готов. Сетевой и локальный retrieval выбираются автоматически.")
        else:
            st.warning("Backend не запущен. Сохранённые вопросы ниже остаются доступны.")

    if run_question:
        mode = "live" if mode_label == "Ответ с доказательствами" else "retrieval_only"
        try:
            with st.spinner("Ищу доказательства в отчёте..."):
                st.session_state["live_result"] = ask_backend(question, mode)
        except requests.RequestException as error:
            st.session_state["live_result"] = None
            st.error(f"Не удалось выполнить запрос: {type(error).__name__}")

    live_result = st.session_state.get("live_result")
    if live_result:
        st.markdown("#### Результат свободного запроса")
        qc = live_result.get("citation_qc", {})
        if qc.get("status") == "pass":
            st.success(live_result["answer"])
        else:
            st.warning(live_result["answer"])
        cache_label = " · кэш: `да`" if live_result.get("cache_hit") else ""
        st.caption(
            f"Источник: `{live_result.get('source')}` · retrieval: "
            f"`{live_result.get('retrieval_transport', 'saved')}` · "
            f"citation QC: `{qc.get('status', 'unknown')}` · модель: "
            f"`{live_result.get('model') or 'не использовалась'}`{cache_label}"
        )
        for ch in live_result.get("evidence", []):
            pages = ", ".join(
                sources[e]["label"] if e in sources else e for e in ch["evidence"]
            )
            with st.expander(
                f"Ранг {ch['rank']} · similarity {ch['similarity']:.3f} · {pages}"
            ):
                st.text(ch["excerpt"][:2500])
                source_buttons(ch["evidence"], key_prefix=f"live_r{ch['rank']}")

    st.divider()
    st.markdown("#### Проверенные примеры")
    presets = B["preset_queries"]
    for i, q in enumerate(presets):
        if st.button(q["question"], key=f"q{i}", use_container_width=True):
            st.session_state["active_q"] = i
    qi = st.session_state.get("active_q")
    if qi is not None and qi < len(presets):
        q = presets[qi]
        st.markdown("#### Ответ")
        st.success(q["answer"])
        st.markdown("#### Найденные фрагменты")
        for ch in q["evidence"]:
            pages = ", ".join(sources[e]["label"] if e in sources else e
                              for e in ch["evidence"])
            with st.expander(f"Ранг {ch['rank']} · similarity {ch['similarity']:.3f} · "
                             f"SOURCE_PAGE: {pages}"):
                st.text(ch["excerpt"][:2500])
                source_buttons(ch["evidence"], key_prefix=f"q{qi}r{ch['rank']}")

# ---------------------------------------------------------------- 3. Сущности
with tabs[2]:
    sub = st.tabs(["Структуры", "Скважины", "Горизонты"])
    ents = B["entities"]

    with sub[0]:
        structs = ents["structures"]
        st.dataframe(pd.DataFrame([{
            "Структура": s["name"], "Статус": s["status"],
            "Горизонты": ", ".join(s["horizons"]) or "—",
            "Confidence": s["confidence"], "Верификация": s["verification"],
            "Evidence": ", ".join(s["evidence"]),
        } for s in structs]), width="stretch", hide_index=True)
        pick = st.selectbox("Детали структуры", [s["name"] for s in structs])
        s = next(x for x in structs if x["name"] == pick)
        st.markdown(f"**{s['name']}** — статус `{s['status']}`, горизонты: "
                    f"{', '.join(s['horizons']) or '—'}, confidence {s['confidence']}, "
                    f"верификация: `{s['verification']}`")
        source_buttons(s["evidence"], key_prefix=f"st_{s['id']}")

    with sub[1]:
        wells = ents["wells"]
        st.dataframe(pd.DataFrame([{
            "Скважина": w["name"], "Площадь": w["area"],
            "Испытания": w["test_result"],
            "Газопроявления": "да" if w["gas_shows"] else "нет указания",
            "Верификация": w["verification"],
        } for w in wells]), width="stretch", hide_index=True)
        pick = st.selectbox("Детали скважины", [w["name"] for w in wells])
        w = next(x for x in wells if x["name"] == pick)
        st.markdown(f"**Скв. {w['name']}** ({w['area']} площадь)")
        st.markdown("- Вскрыто: " + ", ".join(w["penetrated_intervals"]))
        st.markdown(f"- Испытания: {w['test_result']}; газопроявления: "
                    f"{'отмечались' if w['gas_shows'] else 'нет указания'}")
        st.caption(f"Верификация: `{w['verification']}` · модель {w['model']}")
        source_buttons(w["evidence"], key_prefix=f"w_{w['id']}")

    with sub[2]:
        for h in ents["horizons"]:
            with st.expander(f"Горизонт {h['name']} · confidence {h['confidence']} · "
                             f"{h['verification']}", expanded=True):
                for f in h["facts"]:
                    st.markdown(f"- {f}")
                source_buttons(h["evidence"], key_prefix=f"h_{h['id']}")

# ---------------------------------------------------------------- 4. Карта
with tabs[3]:
    mm = B["map_metrics"]
    st.subheader(f"Лист {mm['sheet']}: структурная карта по горизонту К")
    st.warning(
        "**EXPERIMENTAL.** Координаты не геопривязаны "
        f"(`{mm['coordinate_system']}`, georeferenced={mm['georeferenced']}). "
        "Автоматическая реконструкция не прошла ручную валидацию. "
        "Материал иллюстрирует пайплайн и не является финальной картой."
    )
    art = next(a for a in artifacts if a["label"] == map_layer)
    img = load_image(art["path"])
    if img is None:
        st.error("Файл слоя не найден: " + art["path"])
    else:
        st.image(img, caption=f"{art['label']} · reliability: {art['reliability']}",
                 width="stretch")
    for wtext in art["warnings"]:
        st.caption(f"⚠️ {wtext}")
    st.divider()
    c = st.columns(4)
    c[0].metric("Найдено боксов", mm["detected_boxes"])
    c[1].metric("Прочитано", mm["readable_boxes"])
    c[2].metric("Нечитаемых", mm["unreadable_boxes"])
    c[3].metric("Время (baseline)", f"{mm['wall_time_minutes_estimate']} мин")
    st.caption(f"Статус ветки: `{mm['status']}` · {mm['timing_source']}")

# ---------------------------------------------------------------- 5. Агенты
with tabs[4]:
    st.subheader("Цепочка обработки")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fa",
                    fontname="sans-serif", fontsize=11];
  A [label="Page router"];
  B [label="PaddleOCR"];
  C [label="RAGFlow"];
  D [label="Structure agent"];
  E [label="Well agent"];
  F [label="Map agent\\n(experimental)", fillcolor="#fdf2e3"];
  G [label="Evidence QC", fillcolor="#e9f5ec"];
  A -> B -> C; C -> D; C -> E; A -> F; D -> G; E -> G; F -> G;
}""")
    details = {
        "page-router": {
            "in": "Сканы оглавлений томов; список графических приложений.",
            "out": "Типизация листов: схема профилей (13), разрезы (15-19), карты "
                   "изохрон (20, 22), структурная карта (21); router_confidence 0.95.",
            "src": "sources: graphics:13…22 в бандле",
        },
        "paddle-ocr": {
            "in": f"{summary['text_pages_ocr']} страниц текста тома 1.",
            "out": "Постраничный текст с маркерами SOURCE_PAGE (provenance-корпус).",
            "src": "full_report_ocr/report_384092_text_full_*.md",
        },
        "ragflow": {
            "in": "Provenance-корпус.",
            "out": f"{summary['rag_chunks']} чанков, {summary['rag_tokens']:,} токенов; "
                   "топ-5 фрагментов на вопрос (вкладка «Поиск»).",
            "src": "preset_queries в бандле",
        },
        "structure-agent": {
            "in": "Фрагменты RAGFlow по структурам/горизонтам.",
            "out": f"{summary['structures']} структуры, {summary['horizons']} горизонта, "
                   f"{len(B['entities']['survey_tasks'])} задачи; verification=rag_citation_qc.",
            "src": "entities.structures / horizons / survey_tasks",
        },
        "well-agent": {
            "in": "Фрагменты по скважинам + вырезки исходных сканов стр. 24-25.",
            "out": "Скважины 401, 410, 1-П; verification=source_image_vlm; разрешена "
                   "OCR-неоднозначность «1-П».",
            "src": "entities.wells (модель gpt-4.1-mini)",
        },
        "map-agent": {
            "in": "Лист 21 (структурная карта, 1:200000).",
            "out": f"{mm['detected_boxes']} боксов -> {mm['readable_boxes']} прочитано; "
                   f"статус `{mm['status']}`; см. вкладку «Карта».",
            "src": "artifacts map:21:* в бандле",
        },
        "evidence-qc": {
            "in": "Выходы всех агентов.",
            "out": "Каждый факт снабжён evidence-источником; для карты — только "
                   "внутренняя согласованность (не ground truth). Статус partial.",
            "src": "limitations в бандле",
        },
    }
    for stage in B["pipeline"]:
        d = details.get(stage["id"], {})
        icon = STATUS_ICONS.get(stage["status"], "•")
        with st.expander(f"{icon} {stage['label']} — {stage['status']}"
                         + (" · внешний API" if stage["calls_external_api"] else "")):
            st.markdown(f"**Вход:** {d.get('in', '—')}")
            st.markdown(f"**Результат:** {d.get('out', '—')}")
            st.caption(f"Источники: {d.get('src', '—')}")

# ---------------------------------------------------------------- 6. QC
with tabs[5]:
    st.subheader("Контроль качества")
    st.markdown("#### Верифицированные скважины (по исходным сканам)")
    for w in B["entities"]["wells"]:
        st.markdown(
            f"- **Скв. {w['name']}** ({w['area']}): {w['test_result']}; газопроявления: "
            f"{'да' if w['gas_shows'] else 'нет указания'} · `{w['verification']}` "
            f"· {w['model']}")
    st.markdown("#### Происхождение данных (provenance)")
    st.markdown(
        "- Текст: PaddleOCR, постранично, маркеры SOURCE_PAGE в каждом чанке.\n"
        "- Структуры/горизонты/задачи: verification `rag_citation_qc` "
        "(цитаты проверены на валидность страниц).\n"
        "- Скважины: verification `source_image_vlm` — идентификаторы перечитаны "
        "по вырезкам исходного скана.\n"
        "- Карта: см. reliability и warnings у каждого слоя на вкладке «Карта»."
    )
    st.markdown("#### Ограничения картографической ветки")
    for lim in B["limitations"]:
        st.error(f"[{lim['severity']}] {lim['text']}")

# ---------------------------------------------------------------- 7. Автозапуск
with tabs[6]:
    st.subheader("Последний автоматический запуск")
    if not FACTORY_BUNDLE.exists():
        st.info("Результат фабрики ещё не собран.")
    else:
        F = load_factory_bundle(str(FACTORY_BUNDLE), FACTORY_BUNDLE.stat().st_mtime_ns)
        fs = F["summary"]
        st.caption(
            f"Стратегия: `{F['strategy']}` · статус: `{F['status']}` · "
            f"выбрано OCR-страниц: {fs['fast_ocr_pages']} из {fs['page_count']}"
        )
        cols = st.columns(5)
        cols[0].metric("Структуры", fs["structures"])
        cols[1].metric("Скважины", fs["wells"])
        cols[2].metric("Карты и схемы", fs["maps"])
        cols[3].metric("Ключевые результаты", fs["key_results"])
        cols[4].metric("RAG-чанки", fs.get("rag_chunks") or 0)

        map_run = F.get("map_digitization")
        if map_run:
            ms = map_run["summary"]
            with st.expander("Межкартовый тест оцифровки · лист 22", expanded=True):
                mc = st.columns(5)
                mc[0].metric("CV-кандидаты", ms["candidates"])
                mc[1].metric("VLM-запросы", ms["requests"])
                mc[2].metric("Прочитано", ms["readable"])
                mc[3].metric("QC", ms["qc_review"])
                mc[4].metric("Медиана", ms["numeric_median"])
                digitized = map_run.get("digitized")
                if digitized:
                    tracing = map_run.get("tracing")
                    stitching = map_run.get("stitching")
                    st.caption(
                        f"Принято измерений: {digitized['accepted_points']} · "
                        f"координаты: `{digitized['coordinate_system']}` · "
                        f"геопривязка: `{digitized['georeferenced']}`"
                    )
                    if tracing:
                        st.caption(
                            f"Трассировано полилиний: {tracing['vector_polylines']} · "
                            f"подписей изохрон: {tracing['isoline_value_labels']} · "
                            f"автопривязано: {tracing['assigned_value_labels']}"
                        )
                    if stitching:
                        st.caption(
                            f"Топология: {stitching['raw_vector_fragments']} фрагментов → "
                            f"{stitching['vector_polylines']} цепочек · "
                            f"якорей значений: {stitching['assigned_labels']}/{stitching['value_labels']} · "
                            f"рабочих атрибутированных фрагментов: {stitching['work_polylines']} · "
                            f"конфликтов компонент: {stitching['component_value_conflicts']}"
                        )
                    tab_names = ["CV-детекция", "Измерения"]
                    map_paths = [
                        map_run["overlay_path"],
                        digitized["measured_points_overlay"],
                    ]
                    map_captions = [
                        "Переносимый CV-детектор",
                        "Мелкие отметки времени после форматного и пространственного QC",
                    ]
                    if tracing:
                        tab_names.append("Изолинии с растра")
                        map_paths.append(tracing["overlay"])
                        map_captions.append("Красным: трассированные кривые; синим: удалённые прямые профили")
                    if stitching:
                        tab_names.append("Карта на grid")
                        map_paths.append(stitching["grid_map"])
                        map_captions.append(
                            "Самостоятельная цифровая карта в пиксельных координатах; геопривязка пока не применена"
                        )
                        tab_names.append("Рабочая карта")
                        map_paths.append(stitching["attributed_overlay"])
                        map_captions.append(
                            "Цветом: линии с подтверждённым значением; серым: найденная геометрия без надёжной атрибуции"
                        )
                        tab_names.append("Сшивка топологии")
                        map_paths.append(stitching["overlay"])
                        map_captions.append(
                            "Красным: реальные линии растра; пурпурным: восстановленные разрывы по подписи и касательной"
                        )
                    tab_names.append("QC-интерполяция")
                    map_paths.append(digitized["point_interpolation_qc"])
                    map_captions.append("Диагностическая интерполяция измерений; не является оцифрованной картой")
                    map_tabs = st.tabs(tab_names)
                    for map_tab, map_path, map_caption in zip(map_tabs, map_paths, map_captions):
                        with map_tab:
                            image = load_image(map_path)
                            if image is not None:
                                st.image(image, caption=map_caption, width="stretch")
                    if stitching:
                        export_columns = st.columns(2)
                        all_lines_path = Path(stitching["vector_geojson"])
                        work_lines_path = Path(stitching["work_vector_geojson"])
                        if all_lines_path.exists():
                            export_columns[0].download_button(
                                "Скачать все трассы GeoJSON",
                                data=all_lines_path.read_bytes(),
                                file_name="sheet_22_all_traced_isolines.geojson",
                                mime="application/geo+json",
                            )
                        if work_lines_path.exists():
                            export_columns[1].download_button(
                                "Скачать рабочие изолинии GeoJSON",
                                data=work_lines_path.read_bytes(),
                                file_name="sheet_22_attributed_isolines.geojson",
                                mime="application/geo+json",
                            )
                else:
                    overlay = load_image(map_run["overlay_path"])
                    if overlay is not None:
                        st.image(overlay, caption="Переносимый CV-детектор на другом листе", width="stretch")

        entity_tabs = st.tabs(["Структуры", "Скважины", "Карты", "Результаты", "QC"])
        entity_names = ["structures", "wells", "maps", "key_results"]
        for tab, name in zip(entity_tabs[:4], entity_names):
            with tab:
                items = F["entities"].get(name, [])
                if items:
                    st.dataframe(pd.json_normalize(items), width="stretch", hide_index=True)
                else:
                    st.caption("Сущности этого типа не найдены.")
        with entity_tabs[4]:
            if F["verification_queue"]:
                st.warning("Есть элементы для точечной проверки исходных страниц.")
                st.json(F["verification_queue"])
            else:
                st.success("Все ссылки на источники прошли автоматический контроль.")
            st.markdown("**Использованные страницы**")
            for src in F["sources"]:
                label = src["relative_path"] or src["id"]
                if st.button(label, key=f"factory_{src['id']}", disabled=not src["exists"]):
                    st.session_state["page_viewer"] = src["path"]
                    st.session_state["page_viewer_label"] = label
                    st.rerun()
