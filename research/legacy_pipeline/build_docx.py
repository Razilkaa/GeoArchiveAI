"""Build conference theses docx in the format of Тезисы_Киямов.Р.Р.pdf
(RU version + full EN version)."""
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Cm, Pt

OUT = r"C:\FINAM\Conference\docs\theses\Тезисы_Киямов_v3.docx"
FIG = r"C:\FINAM\Conference\figures"

doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(12)
st.paragraph_format.space_after = Pt(0)
for sec in doc.sections:
    sec.top_margin = sec.bottom_margin = Cm(2)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(1.5)


def para(align="justify", indent=True):
    p = doc.add_paragraph()
    p.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
                   "center": WD_ALIGN_PARAGRAPH.CENTER,
                   "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
    if indent:
        p.paragraph_format.first_line_indent = Cm(1.25)
    return p


def text_p(t, bold=False, italic=False, align="justify", indent=True):
    p = para(align, indent)
    r = p.add_run(t)
    r.bold = bold
    r.italic = italic
    return p


def authors_line(items):
    p = para("center", False)
    for k, (name, sup, ul) in enumerate(items):
        if k:
            p.add_run(", ").bold = True
        r = p.add_run(name)
        r.bold = r.italic = True
        r.underline = ul
        rs = p.add_run(sup)
        rs.bold = rs.italic = True
        rs.font.superscript = True
    return p


def picture(path, width_cm, caption):
    p = para("center", False)
    p.add_run().add_picture(path, width=Cm(width_cm))
    text_p(caption, align="center", indent=False)


def header_block(title, authors, affils, email, phone):
    text_p(title, bold=True, align="center", indent=False)
    authors_line(authors)
    for a in affils:
        text_p(a, align="center", indent=False)
    text_p(email, align="center", indent=False)
    text_p(phone, align="center", indent=False)
    doc.add_paragraph()


# ================= RU =================
header_block(
    "Автоматизация анализа архивных геолого-геофизических отчётов с применением "
    "больших языковых моделей",
    [("Р.Р. Киямов", "1,2", True)],
    ["1 – Российский государственный университет нефти и газа (национальный "
     "исследовательский университет) имени И.М. Губкина",
     "2 – АО «Инвестиционная компания «ФИНАМ»"],
    "E-mail: [e-mail]", "Телефон: [телефон]")

text_p("Аннотация", bold=True)
text_p("Значительная часть геологической информации по изученным районам хранится в фондовых "
       "отчётах в виде сканированных документов: тексты, таблицы, структурные карты и разрезы "
       "недоступны для программной обработки. В работе предложена архитектура автоматизации "
       "анализа таких материалов на основе совместного применения классических алгоритмов "
       "компьютерного зрения, оптического распознавания символов (OCR) и больших языковых "
       "моделей (LLM). Языковая модель не заменяет вычислительные методы, а выполняет "
       "функции, ранее требовавшие участия эксперта: исправление распознанного текста, "
       "классификацию объектов, перекрёстный контроль качества между текстом отчёта и "
       "графическими приложениями. Подход апробирован на отчёте сейсморазведочной партии "
       "1980 г.: автоматически распознано оглавление графических приложений, на структурной "
       "карте выделено более 1000 текстовых объектов. Результаты экспортируются в открытые "
       "форматы GeoJSON и GeoPackage. Все компоненты системы допускают развёртывание на "
       "локальной инфраструктуре с использованием моделей с открытыми весами.")
text_p("Ключевые слова", bold=True)
text_p("большие языковые модели, RAG, автоматизация, архивные отчёты, OCR, компьютерное "
       "зрение, структурные карты, GeoPackage")
doc.add_paragraph()

text_p("Фонды геологической информации содержат десятки тысяч отчётов о геологоразведочных "
       "работах прошлых десятилетий. Для территорий с высокой степенью изученности эти "
       "материалы нередко являются единственным источником сведений о строении разреза, однако "
       "существуют только в виде сканов: поиск ведётся вручную, а количественные данные "
       "(отметки горизонтов, результаты испытаний, параметры съёмок) повторно оцифровываются "
       "при каждом обращении. Развитие больших языковых и мультимодальных моделей делает "
       "автоматизацию работы с такими документами практически достижимой.")
text_p("Прямое применение языковых моделей к сканам карт показывает низкую точность "
       "локализации объектов. Классические системы векторизации, напротив, не справляются с "
       "семантикой: разрывы линий на подписях, наложение объектов, рукописные чертёжные "
       "шрифты. В предлагаемом решении функции разделены (рисунок 1): детекция текстовых "
       "областей, линий и засечек выполняется детерминированными алгоритмами компьютерного "
       "зрения (OpenCV); чтение вырезанных фрагментов, классификация и исправление ошибок "
       "распознавания выполняются языковыми моделями; контроль качества основан на правилах "
       "и статистике листа.")
picture(FIG + r"\fig1_architecture.png", 15.5, "Рисунок 1 – Архитектура системы автоматизации")

text_p("Обработка отчёта начинается с оглавления графических приложений. OCR-модель "
       "(PaddleOCR) извлекает текстовые блоки и структуру таблицы (рисунок 2): номера "
       "приложений, наименования, гриф и привязка к томам восстанавливаются программно [1]. "
       "Характерные ошибки распознавания машинописи (замены I/1 и О/0, пропуски символов в "
       "местах дефектов бумаги) исправляются языковой моделью по контексту строки. Далее "
       "агент-маршрутизатор типизирует приложения (карта, схема, разрез, таблица) и направляет "
       "каждый лист в соответствующий модуль обработки. Для апробируемого отчёта система "
       "автоматически определила лист структурной карты и схему расположения профилей.")
picture(FIG + r"\fig2_ocr_toc.jpg", 16.0,
        "Рисунок 2 – Распознавание оглавления графических приложений: слева скан с детекцией "
        "текстовых областей, справа восстановленная структура таблицы")

text_p("Тексты отчёта индексируются в базе знаний с гибридным поиском и моделью "
       "переранжирования (retrieval-augmented generation). Это обеспечивает ответы на вопросы "
       "по содержанию отчёта со ссылками на страницы и используется модулем контроля качества "
       "как источник эталонных сведений: число и названия структур, глубины горизонтов, "
       "параметры съёмки [2].")
text_p("Для графических приложений применяется трёхуровневая схема. По статистике листа "
       "(распределению высот символов) выделяются текстовые области разных классов шрифта; на "
       "апробируемой структурной карте выделено 1088 объектов (рисунок 3). Мультимодальная "
       "модель читает вырезанные фрагменты, классификация выполняется по формату значения: "
       "отметки пикетов, подписи изогипс, номера сейсмопрофилей, скважины. Контроль качества "
       "сопоставляет извлечённые значения с сечением изогипс и данными текста отчёта; каждое "
       "замечание снабжается ссылкой на страницу и фрагментом изображения.")
picture(FIG + r"\fig2_detection.png", 13.0,
        "Рисунок 3 – Фрагмент структурной карты: автоматическая детекция отметок пикетов "
        "(красные рамки) и подписей изогипс (зелёные рамки)")

text_p("Результаты извлечения экспортируются в форматы GeoJSON и GeoPackage и загружаются в "
       "геоинформационные системы и пакеты интерпретации [3]. Поскольку фондовые материалы, "
       "как правило, не могут передаваться во внешние облачные сервисы, все компоненты "
       "системы (OCR, языковая и мультимодальная модели, база знаний) развёртываются на "
       "локальной инфраструктуре с использованием моделей с открытыми весами.")
text_p("Предложенная архитектура носит поэтапный характер: каждый модуль (поиск по текстам "
       "отчётов, извлечение табличных данных, обработка карт) имеет самостоятельную "
       "практическую ценность и измеримые метрики качества. Это позволяет внедрять систему "
       "постепенно, начиная с базы знаний по отчётам и заканчивая полной оцифровкой "
       "графических приложений с автоматическим контролем качества.")
doc.add_paragraph()

text_p("Список литературы", bold=True)
for r in [
    "1. Li C., Liu W., Guo R. et al. PP-OCRv3: More Attempts for the Improvement of Ultra "
    "Lightweight OCR System // arXiv:2206.03001. 2022.",
    "2. Lewis P., Perez E., Piktus A. et al. Retrieval-Augmented Generation for "
    "Knowledge-Intensive NLP Tasks // Advances in Neural Information Processing Systems. "
    "2020. Vol. 33. P. 9459–9474.",
    "3. OGC GeoPackage Encoding Standard. Version 1.4.0. Open Geospatial Consortium, 2024.",
]:
    text_p(r, indent=False)

# ================= EN =================
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
header_block(
    "Automation of archival geological and geophysical report analysis using "
    "large language models",
    [("R.R. Kiamov", "1,2", True)],
    ["1 – National University of Oil and Gas «Gubkin University»",
     "2 – FINAM Investment Company"],
    "E-mail: [e-mail]", "Phone number: [phone]")

text_p("Abstract", bold=True)
text_p("A significant part of geological information for well-studied regions is stored in "
       "archival reports as scanned documents: texts, tables, structural maps and sections are "
       "inaccessible to software processing. This paper proposes an architecture for automating "
       "the analysis of such materials based on the joint use of classical computer vision "
       "algorithms, optical character recognition (OCR) and large language models (LLM). The "
       "language model does not replace computational methods; it performs functions that "
       "previously required an expert: correcting recognized text, classifying objects, and "
       "cross-checking quality between the report text and graphical appendices. The approach "
       "was tested on a 1980 seismic survey report: the table of contents of graphical "
       "appendices was recognized automatically, and more than 1000 text objects were detected "
       "on the structural map. The results are exported to the open GeoJSON and GeoPackage "
       "formats. All system components can be deployed on local infrastructure using "
       "open-weight models.")
text_p("Keywords", bold=True)
text_p("large language models, RAG, automation, archival reports, OCR, computer vision, "
       "structural maps, GeoPackage")
doc.add_paragraph()

for t in [
    "Geological information archives contain tens of thousands of exploration reports from "
    "past decades. For well-studied territories, these materials are often the only source of "
    "information about the subsurface structure, yet they exist only as scans: search is "
    "performed manually, and quantitative data (horizon depths, test results, survey "
    "parameters) are re-digitized at each use. The development of large language and "
    "multimodal models makes the automation of such document processing practically "
    "achievable.",
    "Direct application of language models to map scans shows low object localization "
    "accuracy. Classical vectorization systems, on the contrary, fail at semantics: line "
    "breaks at labels, overlapping objects, handwritten drafting fonts. In the proposed "
    "solution the functions are separated (Figure 1): detection of text regions, lines and "
    "tick marks is performed by deterministic computer vision algorithms (OpenCV); reading of "
    "cropped fragments, classification and correction of recognition errors are performed by "
    "language models; quality control is based on rules and sheet statistics.",
    "Report processing starts with the table of contents of graphical appendices. The OCR "
    "model (PaddleOCR) extracts text blocks and the table structure (Figure 2): appendix "
    "numbers, titles, classification labels and volume references are reconstructed "
    "programmatically [1]. Typical typewriting recognition errors (I/1 and O/0 substitutions, "
    "missing characters at paper defects) are corrected by the language model using the line "
    "context. A routing agent then classifies the appendices (map, scheme, section, table) "
    "and sends each sheet to the corresponding processing module.",
    "Report texts are indexed in a knowledge base with hybrid search and a reranking model "
    "(retrieval-augmented generation). This provides answers to questions about the report "
    "content with page references and serves the quality control module as a source of "
    "reference information: the number and names of structures, horizon depths, survey "
    "parameters [2].",
    "A three-level scheme is applied to graphical appendices. Text regions of different font "
    "classes are detected using sheet statistics (character height distribution); 1088 "
    "objects were detected on the studied structural map (Figure 3). A multimodal model "
    "reads the cropped fragments, and classification is performed by value format: picket "
    "depth marks, isoline labels, seismic profile numbers, wells. Quality control compares "
    "the extracted values with the isoline interval and the report text data; each issue is "
    "accompanied by a page reference and an image fragment.",
    "The extraction results are exported to the GeoJSON and GeoPackage formats and loaded "
    "into GIS and interpretation software [3]. Since archival materials generally cannot be "
    "transferred to external cloud services, all system components (OCR, language and "
    "multimodal models, the knowledge base) are deployed on local infrastructure using "
    "open-weight models.",
    "The proposed architecture is incremental: each module (report text search, tabular data "
    "extraction, map processing) has independent practical value and measurable quality "
    "metrics. This allows gradual adoption, starting with a report knowledge base and ending "
    "with full digitization of graphical appendices with automatic quality control.",
]:
    text_p(t)
doc.add_paragraph()

text_p("References", bold=True)
for r in [
    "1. Li C., Liu W., Guo R. et al. PP-OCRv3: More Attempts for the Improvement of Ultra "
    "Lightweight OCR System // arXiv:2206.03001. 2022.",
    "2. Lewis P., Perez E., Piktus A. et al. Retrieval-Augmented Generation for "
    "Knowledge-Intensive NLP Tasks // Advances in Neural Information Processing Systems. "
    "2020. Vol. 33. P. 9459–9474.",
    "3. OGC GeoPackage Encoding Standard. Version 1.4.0. Open Geospatial Consortium, 2024.",
]:
    text_p(r, indent=False)

doc.save(OUT)
print("saved:", OUT)
