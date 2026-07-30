import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowUpRight, CheckCircle2, CircleAlert, Database,
  FileArchive, FileSearch, Layers3, LoaderCircle, Map, Search, Send, Upload, X
} from "lucide-react";
import { api, type MapArtifact } from "./api";

type Tab = "maps" | "rag" | "global";

function basename(value?: string) {
  return (value || "Архивный отчёт").split(/[\\/]/).filter(Boolean).at(-1) || "Архивный отчёт";
}

function StatusDot({ status }: { status?: string }) {
  const value = status || "pending";
  const label = value === "completed" ? "готов"
    : value === "running" ? "в работе"
    : value === "failed" ? "ошибка"
    : value === "queued" ? "в очереди"
    : "ожидает";
  return <span className={`status status-${value}`}><span />{label}</span>;
}

function MapCard({
  artifact,
  title,
}: {
  artifact?: MapArtifact;
  title: string;
}) {
  const imageUrl = artifact?.preview_url || artifact?.download_url;
  return <div className="map-card">
    <div className="map-card-head">
      <span>{title}</span>
      {artifact ? <a href={artifact.download_url} target="_blank" rel="noreferrer">
        Оригинал <ArrowUpRight size={14}/>
      </a> : null}
    </div>
    {artifact && imageUrl ? <div className="map-frame">
      <div className="map-image-wrap">
        <img src={imageUrl} alt={title} loading="lazy" decoding="async"/>
      </div>
    </div> : <div className="empty-map"><Map size={30}/><span>Результат ещё не сформирован</span></div>}
  </div>;
}

function MapsView({ reportId }: { reportId: string }) {
  const maps = useQuery({
    queryKey: ["maps", reportId],
    queryFn: () => api.maps(reportId),
    refetchInterval: 10_000,
  });
  const [page, setPage] = useState("");

  if (maps.isLoading || !maps.data) {
    return <div className="loading"><LoaderCircle className="spin"/>Загружаем карты</div>;
  }

  const data = maps.data;
  const artifactsByPage = new globalThis.Map<string, MapArtifact[]>();
  data.digitized.forEach(item => {
    if (item.page_id) {
      artifactsByPage.set(item.page_id, [...(artifactsByPage.get(item.page_id) || []), item]);
    }
  });
  const digitizedPages = new Set(artifactsByPage.keys());
  const pages = Array.from(new Set(
    data.sources.map(item => item.page_id).filter(Boolean) as string[]
  )).sort((left, right) => {
    const rank = (value: string) => digitizedPages.has(value) ? 0 : 1;
    return rank(left) - rank(right) || left.localeCompare(right, "ru");
  });
  const selected = page && pages.includes(page) ? page : pages[0];
  const source = data.sources.find(item => item.page_id === selected);
  const artifacts = data.digitized.filter(item => item.page_id === selected);
  const preview = artifacts.find(item =>
    ["surface_clean_preview", "surface_preview", "digitized_map", "georef_raster"].includes(item.name || "")
  );
  const grid = artifacts.find(item => item.name === "georef_cps3")
    || artifacts.find(item => item.name === "local_cps3");
  const lines = artifacts.find(
    item => item.name === "georef_digitized_isolines_by_level_cps3_lines"
  );
  const downloads = [
    grid ? {
      ...grid,
      downloadLabel: grid.name === "georef_cps3"
        ? "Привязанный Grid CPS-3"
        : "Grid CPS-3 · локальные координаты",
    } : undefined,
    lines ? {...lines, downloadLabel: "Привязанные линии CPS-3"} : undefined,
  ].filter(Boolean) as (MapArtifact & {downloadLabel: string})[];

  return <div className="maps-view">
    <div className="toolbar">
      <div>
        <span className="eyebrow">Оцифровка карт</span>
        <h2>{digitizedPages.size} обработано из {pages.length}</h2>
        <small className={`map-status map-status-${source?.status || "pending"}`}>
          {source?.status === "accepted" ? "принято"
            : source?.status === "review" ? "нужна проверка"
            : source?.status === "failed" ? "ошибка обработки"
            : source?.status === "not_applicable" ? "не распознано как карта"
            : "ожидает обработки"}
        </small>
      </div>
      <div className="map-toolbar-actions">
        {pages.length ? <select value={selected} onChange={event => setPage(event.target.value)}>
          {pages.map(value => {
            const item = data.sources.find(candidate => candidate.page_id === value);
            const location = (item?.path || "")
              .split(/[\\/]/)
              .filter(Boolean)
              .at(-1) || "";
            const sheet = item?.page_number ? `Лист ${item.page_number}` : value;
            return <option key={value} value={value}>
              {pages.length > 1 && location ? `${sheet} · ${location}` : sheet}
            </option>;
          })}
        </select> : null}
      </div>
    </div>

    {!pages.length ? <div className="empty"><Layers3/><h3>Карты не обнаружены</h3></div> : <>
      <div className="map-grid">
        <MapCard artifact={source} title="Исходник"/>
        <MapCard artifact={preview} title="Результат оцифровки"/>
      </div>
      <div className="downloads">
        {downloads.map(item => <a className="download" key={item.artifact_id} href={item.download_url} download>
          <FileSearch size={18}/><span>{item.downloadLabel}<small>{item.media_type}</small></span><ArrowUpRight size={16}/>
        </a>)}
      </div>
    </>}
  </div>;
}

function RagView({ reportId, global = false }: { reportId: string; global?: boolean }) {
  const [question, setQuestion] = useState("");
  const mutation = useMutation({
    mutationFn: (value: string) => global ? api.search(value) : api.ask(reportId, value),
  });
  const submit = () => {
    if (question.trim()) mutation.mutate(question.trim());
  };
  return <div className="rag-layout">
    <div className="rag-main">
      <span className="eyebrow">{global ? "Поиск по фонду" : "Поиск по отчёту"}</span>
      <h2>{global ? "Спросите весь архивный фонд" : "Спросите этот отчёт"}</h2>
      <p className="muted">Поиск и генерация выполняются через RAGFlow. Справа всегда показаны исходные фрагменты.</p>
      <div className="prompt">
        <textarea
          value={question}
          onChange={event => setQuestion(event.target.value)}
          onKeyDown={event => {
            if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) submit();
          }}
          placeholder="Например: какие структуры рекомендованы для глубокого бурения?"
        />
        <button onClick={submit} disabled={!question.trim() || mutation.isPending}>
          {mutation.isPending ? <LoaderCircle className="spin"/> : <Send/>}
        </button>
      </div>
      {mutation.error ? <div className="error-box"><CircleAlert/>Ошибка поиска: {mutation.error.message}</div> : null}
      {mutation.data ? <article className="answer">
        <div className="answer-meta">
          {mutation.data.citation_qc?.status === "pass" ? <CheckCircle2/> : <CircleAlert/>}
          Черновик ответа · сверяйте с источниками
          <span>{(((mutation.data.retrieval_latency_ms || 0) + (mutation.data.generation_latency_ms || 0)) / 1000).toFixed(1)}s</span>
        </div>
        <div className="answer-text">{mutation.data.answer}</div>
      </article> : null}
    </div>
    <aside className="sources">
      <h3>Источники</h3>
      {mutation.data?.evidence?.map((item, index) => <div className="source" key={index}>
        <div><span>#{index + 1}</span><b>{item.evidence?.join(", ") || "фрагмент"}</b><em>{Math.round((item.similarity || 0) * 100)}%</em></div>
        <p>{item.excerpt}</p>
      </div>)}
      {!mutation.data ? <p className="muted">Здесь появятся найденные фрагменты и их релевантность.</p> : null}
    </aside>
  </div>;
}

function App() {
  const queryClient = useQueryClient();
  const reportsQuery = useQuery({
    queryKey: ["reports"],
    queryFn: api.reports,
    refetchInterval: 10_000,
  });
  const [selected, setSelected] = useState("");
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState<Tab>("maps");
  const [uploadOpen, setUploadOpen] = useState(false);
  const upload = useMutation({
    mutationFn: api.upload,
    onSuccess: async result => {
      await queryClient.invalidateQueries({queryKey: ["reports"]});
      if (result.report_ids[0]) setSelected(result.report_ids[0]);
      setTab("maps");
      setUploadOpen(false);
    },
  });
  const reports = useMemo(() => [...(reportsQuery.data?.reports || [])].sort((left, right) =>
    right.report_id.localeCompare(left.report_id, "ru", {numeric: true})
  ), [reportsQuery.data?.reports]);
  const reportId = selected && reports.some(item => item.report_id === selected)
    ? selected
    : reports[0]?.report_id || "";
  const report = reports.find(item => item.report_id === reportId);
  const filtered = reports.filter(item =>
    `${item.report_id} ${item.source_root || ""}`.toLowerCase().includes(query.toLowerCase())
  );
  const tabs: [Tab, typeof Map, string][] = [
    ["maps", Map, "Карты"],
    ["rag", FileSearch, "По отчёту"],
    ["global", Database, "По всему фонду"],
  ];

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div>GA</div><span><b>GeoArchive</b><small>рабочий MVP</small></span></div>
      <button className="upload-button" onClick={() => setUploadOpen(true)}><Upload size={17}/>Загрузить отчёт</button>
      <div className="sidebar-label"><span>Отчёты</span><b>{reports.length}</b></div>
      <div className="search"><Search size={16}/><input value={query} onChange={event => setQuery(event.target.value)} placeholder="Номер или название"/></div>
      <nav className="report-list">
        {reportsQuery.isLoading ? <div className="loading"><LoaderCircle className="spin"/></div>
          : reportsQuery.error ? <div className="error-box">API недоступен</div>
          : filtered.map(item => <button
            className={item.report_id === reportId ? "active" : ""}
            key={item.report_id}
            onClick={() => {setSelected(item.report_id); setTab("maps");}}
          >
            <FileArchive size={17}/><span><b>{item.report_id}</b><small>{item.summary?.title || basename(item.source_root)}</small></span>
            <StatusDot status={item.worker_status}/>
          </button>)}
      </nav>
      <div className="sidebar-footer"><Database size={15}/>RAGFlow · FastAPI</div>
    </aside>
    <main>
      {report ? <>
        <header className="topbar">
          <div><span className="breadcrumb">Фонд · {report.report_id}</span><h1>{report.summary?.title || basename(report.source_root)}</h1><p>{report.summary?.page_count || 0} страниц</p></div>
          <StatusDot status={report.worker_status}/>
        </header>
        <div className="tabs">{tabs.map(([key, Icon, label]) => <button key={key} className={tab === key ? "active" : ""} onClick={() => setTab(key)}><Icon size={17}/>{label}</button>)}</div>
        <div className="content">{tab === "maps" ? <MapsView reportId={reportId}/> : <RagView reportId={reportId} global={tab === "global"}/>}</div>
      </> : <div className="empty landing"><FileArchive/><h1>Фонд пока пуст</h1><p>Загрузите первый архивный отчёт.</p><button onClick={() => setUploadOpen(true)}>Загрузить отчёт</button></div>}
    </main>
    {uploadOpen ? <div className="modal-backdrop"><div className="modal">
      <button className="modal-close" onClick={() => setUploadOpen(false)}><X/></button>
      <div className="upload-icon"><Upload/></div><span className="eyebrow">Новый материал</span>
      <h2>Добавить архивный отчёт</h2>
      <p>ZIP, PDF или комплект сканов. После загрузки OCR, чанкинг и отправка в RAGFlow запустятся автоматически.</p>
      <label className="dropzone">
        <input type="file" onChange={event => {
          const file = event.target.files?.[0];
          if (file) upload.mutate(file);
        }}/>
        {upload.isPending ? <LoaderCircle className="spin"/> : <Upload/>}
        <b>{upload.isPending ? "Загружаем…" : "Выбрать файл"}</b>
        <small>Файл передаётся через FastAPI</small>
      </label>
      {upload.error ? <div className="error-box">{upload.error.message}</div> : null}
    </div></div> : null}
  </div>;
}

export default App;
