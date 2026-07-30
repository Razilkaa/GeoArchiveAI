export type Stage = { status?: string; detail?: string; error?: string };
export type ReportListItem = {
  report_id: string;
  source_root?: string;
  summary?: { page_count?: number };
  worker_status?: string;
  operator_status?: string;
};
export type ReportStatus = {
  report_id: string;
  source_root?: string;
  summary?: { page_count?: number };
  worker?: { status?: string; stages?: Record<string, Stage> };
};
export type Entity = Record<string, unknown> & { evidence?: string[] };
export type Bundle = {
  report_id?: string;
  status?: string;
  summary?: Record<string, unknown>;
  entities?: Record<string, Entity[]>;
};
export type MapArtifact = {
  artifact_id: string;
  page_id?: string;
  page_number?: number;
  name?: string;
  label?: string;
  path?: string;
  media_type?: string;
  download_url: string;
  preview_url?: string;
  exists?: boolean;
  status?: string;
  content_type?: string;
  quality?: Record<string, unknown>;
};
export type MapsPayload = {
  report_id: string;
  status: string;
  quality_status?: string;
  sources: MapArtifact[];
  digitized: MapArtifact[];
  metrics?: Record<string, unknown>;
  issues?: unknown[];
  structures?: Entity[];
  horizons?: Entity[];
};
export type ProfileCrossing = {
  key: string;
  profiles: [string, string];
  label: string;
  map: [number, number];
};
export type ProfileCrossingsPayload = {
  report_id: string;
  page_id: string;
  target_crs: string;
  crossings: ProfileCrossing[];
  raster_size?: [number, number];
};
export type ProfileAnchor = {
  pixel: [number, number];
  crossing_key: string;
};
export type Evidence = { evidence?: string[]; excerpt?: string; similarity?: number; document_name?: string };
export type Answer = { answer: string; evidence?: Evidence[]; citation_qc?: { status?: string }; retrieval_latency_ms?: number; generation_latency_ms?: number };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  reports: () => request<{ reports: ReportListItem[]; count: number }>("/api/reports"),
  status: (id: string) => request<ReportStatus>(`/api/reports/${encodeURIComponent(id)}/status`),
  bundle: (id: string) => request<Bundle>(`/api/reports/${encodeURIComponent(id)}`),
  maps: (id: string) => request<MapsPayload>(`/api/reports/${encodeURIComponent(id)}/maps`),
  profileCrossings: (id: string, pageId: string) => request<ProfileCrossingsPayload>(
    `/api/reports/${encodeURIComponent(id)}/maps/profile-crossings?page_id=${encodeURIComponent(pageId)}`
  ),
  georeferenceMap: (id: string, pageId: string, anchors: ProfileAnchor[]) => request<Record<string, unknown>>(
    `/api/reports/${encodeURIComponent(id)}/maps/georeference`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ page_id: pageId, anchors })
    }
  ),
  services: () => request<Record<string, unknown>>("/api/services"),
  ask: (id: string, question: string) => request<Answer>(`/api/reports/${encodeURIComponent(id)}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, mode: "live" })
  }),
  search: (question: string) => request<Answer>("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, mode: "live" })
  }),
  upload: async (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<{ report_ids: string[] }>("/api/reports/upload", { method: "POST", body });
  }
};
