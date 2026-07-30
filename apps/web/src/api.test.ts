import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";

describe("FastAPI client", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("loads the report catalog through the HTTP contract", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ reports: [{ report_id: "384092" }], count: 1 })
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await api.reports();

    expect(fetchMock).toHaveBeenCalledWith("/api/reports", undefined);
    expect(result.reports[0].report_id).toBe("384092");
  });

  it("sends RAG mode explicitly", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ answer: "ok" })
    });
    vi.stubGlobal("fetch", fetchMock);

    await api.ask("384092", "Какие структуры выделены?");

    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({
      question: "Какие структуры выделены?",
      mode: "live"
    });
  });
});
