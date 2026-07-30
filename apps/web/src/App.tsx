import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Archive, ArrowUpRight, CheckCircle2, ChevronRight, CircleAlert, Database,
  Crosshair, FileArchive, FileSearch, Layers3, LoaderCircle, Map, Search, Send, Upload, X
} from "lucide-react";
import {
  api, type Bundle, type Entity, type MapArtifact, type ProfileAnchor,
  type ReportListItem
} from "./api";

const stageLabels: Record<string, string> = {
  page_routing: "Разбор страниц", fast_ocr: "OCR", markdown: "Markdown",
  ragflow: "RAGFlow", map_digitization: "Карты", agents: "Сущности", bundle: "Результат"
};

function basename(value?: string) {
  return (value || "Архивный отчёт").split(/[\\/]/).filter(Boolean).at(-1) || "Архивный отчёт";
}

function StatusDot({ status }: { status?: string }) {
  const value = status || "pending";
  return <span className={`status status-${value}`}><span />{value === "completed" ? "готов" : value === "running" ? "в работе" : value === "failed" ? "ошибка" : value === "queued" ? "в очереди" : "ожидает"}</span>;
}

function EntityTable({ title, rows }: { title: string; rows: Entity[] }) {
  const keys = useMemo(() => {
    const preferred = ["name", "id", "status", "description", "depth_m", "horizons"];
    return preferred.filter(key => rows.some(row => row[key] !== undefined && row[key] !== null)).slice(0, 4);
  }, [rows]);
  if (!rows.length) return null;
  return <section className="panel entity-panel">
    <div className="panel-title"><div><span className="eyebrow">Извлечено</span><h3>{title}</h3></div><span className="count">{rows.length}</span></div>
    <div className="table-wrap"><table><thead><tr>{keys.map(key => <th key={key}>{key === "name" ? "Название" : key === "id" ? "Идентификатор" : key === "status" ? "Статус" : key === "description" ? "Описание" : key}</th>)}</tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index}>{keys.map(key => <td key={key}>{Array.isArray(row[key]) ? (row[key] as unknown[]).join(", ") : String(row[key] ?? "—")}{key === keys[0] && row.evidence?.length ? <small>{row.evidence.join(" · ")}</small> : null}</td>)}</tr>)}</tbody>
    </table></div>
  </section>;
}

type MapMarker = { xPercent: number; yPercent: number; label: string };

function MapCard({
  artifact,
  title,
  onImageClick,
  markers = [],
}: {
  artifact?: MapArtifact;
  title: string;
  onImageClick?: (pixel: [number, number], relative: [number, number]) => void;
  markers?: MapMarker[];
}) {
  return <div className="map-card"><div className="map-card-head"><span>{title}</span>{artifact ? <a href={artifact.download_url} target="_blank" rel="noreferrer">Открыть <ArrowUpRight size={14}/></a> : null}</div>
    {artifact ? <div className={`map-frame ${onImageClick ? "map-frame-clickable" : ""}`}><div className="map-image-wrap"><img src={artifact.download_url} alt={title} onClick={event => {
      if (!onImageClick) return;
      const rect = event.currentTarget.getBoundingClientRect();
      const relative: [number, number] = [
        (event.clientX - rect.left) / rect.width,
        (event.clientY - rect.top) / rect.height,
      ];
      onImageClick(
        [
          relative[0] * event.currentTarget.naturalWidth,
          relative[1] * event.currentTarget.naturalHeight,
        ],
        relative,
      );
    }}/>{markers.map((marker, index) => <span className="map-anchor-marker" key={`${marker.label}-${index}`} style={{left: `${marker.xPercent}%`, top: `${marker.yPercent}%`}}>{index + 1}</span>)}</div></div> : <div className="empty-map"><Map size={30}/><span>Артефакт ещё не сформирован</span></div>}
  </div>;
}

function MapsView({ reportId }: { reportId: string }) {
  const queryClient = useQueryClient();
  const { data, isLoading } = useQuery({ queryKey: ["maps", reportId], queryFn: () => api.maps(reportId), refetchInterval: 10_000 });
  const [page, setPage] = useState<string>("");
  const [anchorPage, setAnchorPage] = useState<string>("");
  const [georefOpen, setGeorefOpen] = useState(false);
  const [crossingKey, setCrossingKey] = useState("");
  const [anchors, setAnchors] = useState<(ProfileAnchor & MapMarker & { label: string })[]>([]);
  const crossingsQuery = useQuery({
    queryKey: ["profile-crossings", reportId, anchorPage],
    queryFn: () => api.profileCrossings(reportId, anchorPage),
    enabled: georefOpen && Boolean(anchorPage),
    retry: false,
  });
  const georeference = useMutation({
    mutationFn: () => api.georeferenceMap(
      reportId,
      anchorPage,
      anchors.map(item => ({pixel: item.pixel, crossing_key: item.crossing_key})),
    ),
    onSuccess: async () => {
      await queryClient.invalidateQueries({queryKey: ["maps", reportId]});
      setGeorefOpen(false);
      setAnchors([]);
      setCrossingKey("");
    },
  });
  if (isLoading || !data) return <div className="loading"><LoaderCircle className="spin"/>Загружаем карты</div>;
  const artifactsByPage = new globalThis.Map<string, MapArtifact[]>();
  data.digitized.forEach(item => {
    if (!item.page_id) return;
    artifactsByPage.set(item.page_id, [...(artifactsByPage.get(item.page_id) || []), item]);
  });
  const processedArtifactNames = new Set(["surface_clean_preview", "surface_preview", "digitized_map", "georef_delivery", "georef_package"]);
  const pageRank = (pageId: string) => {
    const pageArtifacts = artifactsByPage.get(pageId) || [];
    if (pageArtifacts.some(item => processedArtifactNames.has(item.name || ""))) return 0;
    const status = data.sources.find(item => item.page_id === pageId)?.status;
    return status === "review" ? 1 : status === "failed" ? 3 : status === "not_applicable" ? 2 : 4;
  };
  const pages = (Array.from(new Set(data.sources.map(item => item.page_id).filter(Boolean))) as string[])
    .sort((left, right) => pageRank(left) - pageRank(right) || left.localeCompare(right, "ru"));
  const selected = page && pages.includes(page) ? page : pages[0];
  const source = data.sources.find(item => item.page_id === selected);
  const artifacts = data.digitized.filter(item => item.page_id === selected);
  const preview = artifacts.find(item => ["surface_clean_preview", "surface_preview", "digitized_map", "georef_raster"].includes(item.name || ""));
  const hasGeoreference = artifacts.some(item => ["georef_delivery", "georef_package"].includes(item.name || ""));
  const isGeoreferencedOnly = hasGeoreference && !artifacts.some(item => ["surface_clean_preview", "surface_preview", "digitized_map"].includes(item.name || ""));
  const cps3 = artifacts.find(item => item.name === "georef_cps3") || artifacts.find(item => item.name === "local_cps3");
  const digitizedDownload = artifacts.find(item => item.name === "georef_delivery")
    || artifacts.find(item => item.name === "georef_package")
    || artifacts.find(item => ["surface_clean_preview", "surface_preview", "digitized_map", "georef_raster"].includes(item.name || ""));
  const downloads = [
    cps3 ? { ...cps3, downloadLabel: "CPS-3" } : undefined,
    digitizedDownload ? { ...digitizedDownload, downloadLabel: "Оцифрованный результат" } : undefined,
  ].filter(Boolean) as (MapArtifact & { downloadLabel: string })[];
  const mapStatus = isGeoreferencedOnly ? "Привязано по двум профилям" : source?.status === "accepted" ? "Оцифровано" : source?.status === "review" ? "Требует проверки" : source?.status === "failed" ? "Ошибка обработки" : source?.status === "not_applicable" ? "Исходный лист" : "Ожидает обработки";
  const crossingOptions = crossingsQuery.data?.crossings || [];
  const selectedCrossingKey = crossingKey || crossingOptions.find(item => !anchors.some(anchor => anchor.crossing_key === item.key))?.key || "";
  const selectPage = (value: string) => {
    setPage(value);
    setGeorefOpen(false);
    setAnchorPage("");
    setAnchors([]);
    setCrossingKey("");
  };
  const beginGeoreference = () => {
    setPage(selected);
    setAnchorPage(selected);
    setAnchors([]);
    setCrossingKey("");
    setGeorefOpen(true);
  };
  const placeAnchor = (pixel: [number, number], relative: [number, number]) => {
    const crossing = crossingOptions.find(item => item.key === selectedCrossingKey);
    if (!crossing || anchors.some(item => item.crossing_key === crossing.key)) return;
    setAnchors(current => [...current, {
      pixel,
      crossing_key: crossing.key,
      label: crossing.label,
      xPercent: relative[0] * 100,
      yPercent: relative[1] * 100,
    }]);
    setCrossingKey("");
  };
  return <div className="maps-view">
    <div className="toolbar"><div><span className="eyebrow">Картографические материалы</span><h2>{pages.length} листов</h2><small className={`map-status map-status-${source?.status || "pending"}`}>{mapStatus}</small></div><div className="map-toolbar-actions">{source?.content_type === "map" ? <button className="secondary-button" onClick={beginGeoreference}><Crosshair size={15}/>Привязать по 2 точкам</button> : null}{pages.length ? <select value={selected} onChange={e => selectPage(e.target.value)}>{pages.map(value => { const item = data.sources.find(sourceItem => sourceItem.page_id === value); return <option key={value} value={value}>{value} · {item?.content_type === "map" ? "карта" : "графика"}</option>; })}</select> : null}</div></div>
    {georefOpen ? <div className="georef-panel"><div><span className="eyebrow">Двухточечная привязка</span><b>Выберите пересечение профилей и кликните его точное место на исходном листе</b><small>{crossingsQuery.data ? `CRS: ${crossingsQuery.data.target_crs}` : "Читаем номера профилей и шейп…"}</small></div>{crossingsQuery.isLoading ? <LoaderCircle className="spin"/> : crossingOptions.length ? <><select value={selectedCrossingKey} onChange={event => setCrossingKey(event.target.value)}>{crossingOptions.map(item => <option value={item.key} key={item.key}>{item.label}</option>)}</select><div className="georef-anchors">{anchors.map((item, index) => <span key={item.crossing_key}>{index + 1}. {item.label}</span>)}</div><button disabled={anchors.length < 2 || georeference.isPending} onClick={() => georeference.mutate()}>{georeference.isPending ? <LoaderCircle className="spin"/> : <Crosshair/>}Привязать</button><button className="text-button" onClick={() => {setGeorefOpen(false); setAnchors([]);}}>Отмена</button></> : <span className="error-box">На листе не найдены пересекающиеся номера профилей.</span>}{crossingsQuery.error ? <span className="error-box">{crossingsQuery.error.message}</span> : null}{georeference.error ? <span className="error-box">{georeference.error.message}</span> : null}</div> : null}
    {!pages.length ? <div className="empty"><Layers3/><h3>Карты не обнаружены</h3></div> : <><div className="map-grid"><MapCard artifact={source} title={georefOpen ? "Исходный лист · поставьте якорь" : "Исходный лист"} onImageClick={georefOpen ? placeAnchor : undefined} markers={anchors}/><MapCard artifact={preview} title="Результат оцифровки"/></div>
      {data.structures?.length ? <div className="map-structures"><span className="eyebrow">Перспективные структуры из текста отчёта</span><div>{data.structures.map((item, index) => <span key={index}>{String(item.name || item.title || `Структура ${index + 1}`)}</span>)}</div></div> : null}
      <div className="downloads">{downloads.map(item => <a className="download" key={item.artifact_id} href={item.download_url} download><FileSearch size={18}/><span>{item.downloadLabel}<small>{item.media_type}</small></span><ArrowUpRight size={16}/></a>)}</div></>}
  </div>;
}

function RagView({ reportId, global = false }: { reportId: string; global?: boolean }) {
  const [question, setQuestion] = useState("");
  const mutation = useMutation({ mutationFn: (value: string) => global ? api.search(value) : api.ask(reportId, value) });
  const submit = () => { if (question.trim()) mutation.mutate(question.trim()); };
  return <div className="rag-layout"><div className="rag-main"><span className="eyebrow">{global ? "Поиск по фонду" : "Evidence assistant"}</span><h2>{global ? "Спросите весь архивный фонд" : "Спросите этот отчёт"}</h2><p className="muted">Ответ строится по RAGFlow и сопровождается страницами-источниками.</p>
    <div className="prompt"><textarea value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit(); }} placeholder="Например: какие структуры рекомендованы для глубокого бурения?"/><button onClick={submit} disabled={!question.trim() || mutation.isPending}>{mutation.isPending ? <LoaderCircle className="spin"/> : <Send/>}</button></div>
    {mutation.error ? <div className="error-box"><CircleAlert/>Не удалось выполнить поиск: {mutation.error.message}</div> : null}
    {mutation.data ? <article className="answer"><div className="answer-meta"><CheckCircle2/>Ответ с проверкой цитат <span>{((mutation.data.retrieval_latency_ms || 0) + (mutation.data.generation_latency_ms || 0)) / 1000}s</span></div><div className="answer-text">{mutation.data.answer}</div></article> : null}
  </div><aside className="sources"><h3>Источники</h3>{mutation.data?.evidence?.map((item, index) => <div className="source" key={index}><div><span>#{index + 1}</span><b>{item.evidence?.join(", ") || "фрагмент"}</b><em>{Math.round((item.similarity || 0) * 100)}%</em></div><p>{item.excerpt}</p></div>)}{!mutation.data ? <p className="muted">Здесь появятся найденные фрагменты и их релевантность.</p> : null}</aside></div>;
}

function Overview({ bundle }: { bundle?: Bundle }) {
  const entities = bundle?.entities || {};
  const groups = Object.entries(entities).filter(([, value]) => Array.isArray(value));
  return <div className="overview"><div className="metrics">{[["Структуры", entities.structures?.length || 0], ["Скважины", entities.wells?.length || 0], ["Горизонты", entities.horizons?.length || 0], ["Результаты", entities.key_results?.length || 0]].map(([label, value]) => <div className="metric" key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
    {groups.map(([name, rows]) => <EntityTable key={name} title={name === "structures" ? "Структуры" : name === "wells" ? "Скважины" : name === "horizons" ? "Горизонты" : "Ключевые результаты"} rows={rows}/>)}</div>;
}

function Processing({ reportId }: { reportId: string }) {
  const { data } = useQuery({ queryKey: ["status", reportId], queryFn: () => api.status(reportId), refetchInterval: 7_000 });
  const stages = data?.worker?.stages || {};
  return <div className="processing"><div className="processing-head"><div><span className="eyebrow">Pipeline</span><h2>Обработка отчёта</h2></div><StatusDot status={data?.worker?.status}/></div>
    <div className="stage-list">{Object.entries(stageLabels).map(([key, label], index) => { const stage = stages[key]; return <div className={`stage stage-${stage?.status || "pending"}`} key={key}><span className="stage-number">{String(index + 1).padStart(2, "0")}</span><div><b>{label}</b><small>{stage?.detail || stage?.error || (stage?.status === "completed" ? "Этап завершён" : "Ожидает запуска")}</small></div>{stage?.status === "completed" ? <CheckCircle2/> : stage?.status === "running" ? <LoaderCircle className="spin"/> : <ChevronRight/>}</div>; })}</div>
  </div>;
}

function App() {
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useQuery({ queryKey: ["reports"], queryFn: api.reports, refetchInterval: 10_000 });
  const [selected, setSelected] = useState("");
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState("overview");
  const [uploadOpen, setUploadOpen] = useState(false);
  const upload = useMutation({ mutationFn: api.upload, onSuccess: async result => { await queryClient.invalidateQueries({ queryKey: ["reports"] }); if (result.report_ids[0]) setSelected(result.report_ids[0]); setUploadOpen(false); } });
  const reports = useMemo(() => [...(data?.reports || [])].sort((left, right) => {
    const leftNumber = Number.parseInt(left.report_id, 10);
    const rightNumber = Number.parseInt(right.report_id, 10);
    if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) return rightNumber - leftNumber;
    if (Number.isFinite(leftNumber)) return -1;
    if (Number.isFinite(rightNumber)) return 1;
    return left.report_id.localeCompare(right.report_id, "ru");
  }), [data?.reports]);
  const reportId = selected && reports.some(item => item.report_id === selected) ? selected : reports[0]?.report_id || "";
  const report = reports.find(item => item.report_id === reportId);
  const filtered = reports.filter(item => `${item.report_id} ${item.source_root || ""}`.toLowerCase().includes(query.toLowerCase()));
  const bundle = useQuery({ queryKey: ["bundle", reportId], queryFn: () => api.bundle(reportId), enabled: Boolean(reportId), retry: false });

  return <div className="app-shell"><aside className="sidebar"><div className="brand"><div>GA</div><span><b>GeoArchive</b><small>геологический фонд</small></span></div><button className="upload-button" onClick={() => setUploadOpen(true)}><Upload size={17}/>Новый архив</button>
    <div className="sidebar-label"><span>Отчёты</span><b>{reports.length}</b></div><div className="search"><Search size={16}/><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Номер или название"/></div>
    <nav className="report-list">{isLoading ? <div className="loading"><LoaderCircle className="spin"/></div> : error ? <div className="error-box">API недоступен</div> : filtered.map(item => <button className={item.report_id === reportId ? "active" : ""} key={item.report_id} onClick={() => { setSelected(item.report_id); setTab("overview"); }}><FileArchive size={17}/><span><b>{item.report_id}</b><small>{basename(item.source_root)}</small></span><StatusDot status={item.worker_status}/></button>)}</nav>
    <div className="sidebar-footer"><Database size={15}/>RAGFlow · FastAPI</div></aside>
    <main>{report ? <><header className="topbar"><div><span className="breadcrumb">Архивный фонд <ChevronRight size={14}/> {report.report_id}</span><h1>{basename(report.source_root)}</h1><p>{report.report_id} · {report.summary?.page_count || 0} страниц</p></div><StatusDot status={report.worker_status}/></header>
      <div className="tabs">{[["overview", Archive, "Обзор"], ["processing", LoaderCircle, "Обработка"], ["maps", Map, "Карты"], ["rag", FileSearch, "Поиск"], ["global", Database, "По фонду"]].map(([key, Icon, label]) => { const TabIcon = Icon as typeof Archive; return <button key={key as string} className={tab === key ? "active" : ""} onClick={() => setTab(key as string)}><TabIcon size={17}/>{label as string}</button>; })}</div>
      <div className="content">{tab === "overview" ? <Overview bundle={bundle.data}/> : tab === "processing" ? <Processing reportId={reportId}/> : tab === "maps" ? <MapsView reportId={reportId}/> : tab === "rag" ? <RagView reportId={reportId}/> : <RagView reportId={reportId} global/>}</div></> : <div className="empty landing"><Archive/><h1>Фонд пока пуст</h1><p>Загрузите первый архивный отчёт.</p><button onClick={() => setUploadOpen(true)}>Загрузить архив</button></div>}</main>
    {uploadOpen ? <div className="modal-backdrop"><div className="modal"><button className="modal-close" onClick={() => setUploadOpen(false)}><X/></button><div className="upload-icon"><Upload/></div><span className="eyebrow">Новый материал</span><h2>Добавить архивный отчёт</h2><p>ZIP, PDF или комплект сканов. После загрузки обработка запустится автоматически.</p><label className="dropzone"><input type="file" onChange={e => { const file = e.target.files?.[0]; if (file) upload.mutate(file); }}/>{upload.isPending ? <LoaderCircle className="spin"/> : <Upload/>}<b>{upload.isPending ? "Загружаем…" : "Выбрать файл"}</b><small>Файл будет передан FastAPI</small></label>{upload.error ? <div className="error-box">{upload.error.message}</div> : null}</div></div> : null}
  </div>;
}

export default App;
