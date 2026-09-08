// Tests for Voyage contextualized-chunk embedding requests and parsing.
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  embedVoyageContextualized,
  isVoyageContextualizedModel,
  VOYAGE_CONTEXTUALIZED_CHUNK_SIZE,
} from "./contextualized-embedding.js";
import type { VoyageEmbeddingClient } from "./embedding-provider.js";

function buildClient(): VoyageEmbeddingClient {
  return {
    baseUrl: "https://api.voyageai.test/v1",
    headers: { authorization: "Bearer fixture-voyage" },
    model: "voyage-context-4",
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

// Contextualized response: each input maps to a list of chunk embeddings.
function contextualizedResponse(vectors: number[][]): Response {
  return jsonResponse({
    object: "list",
    data: vectors.map((embedding, index) => ({
      object: "list",
      index,
      data: [{ object: "embedding", embedding, index: 0 }],
    })),
    model: "voyage-context-4",
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("isVoyageContextualizedModel", () => {
  it("matches voyage-context models only", () => {
    expect(isVoyageContextualizedModel("voyage-context-4")).toBe(true);
    expect(isVoyageContextualizedModel("voyage-context-3")).toBe(true);
    expect(isVoyageContextualizedModel("voyage-4-large")).toBe(false);
    expect(isVoyageContextualizedModel("voyage-code-4")).toBe(false);
  });
});

describe("embedVoyageContextualized", () => {
  it("calls contextualized_embed with list input, auto-chunking, and full-window chunk size", async () => {
    let capturedUrl = "";
    let capturedBody: Record<string, unknown> = {};
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      capturedUrl = String(input);
      capturedBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return contextualizedResponse([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });
    vi.stubGlobal("fetch", fetchMock);

    const vectors = await embedVoyageContextualized({
      client: buildClient(),
      inputs: ["alpha", "beta"],
      inputType: "document",
    });

    expect(capturedUrl).toBe("https://api.voyageai.test/v1/contextualizedembeddings");
    expect(capturedBody).toMatchObject({
      model: "voyage-context-4",
      inputs: ["alpha", "beta"],
      input_type: "document",
      enable_auto_chunking: true,
      chunk_size: VOYAGE_CONTEXTUALIZED_CHUNK_SIZE,
    });
    expect(Array.isArray(capturedBody.inputs)).toBe(true);
    expect(vectors).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);
  });

  it("omits auto-chunking for query inputs", async () => {
    let capturedBody: Record<string, unknown> = {};
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        capturedBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
        return contextualizedResponse([[7, 8]]);
      }),
    );

    const vectors = await embedVoyageContextualized({
      client: buildClient(),
      inputs: ["query text"],
      inputType: "query",
    });

    expect(capturedBody.input_type).toBe("query");
    expect(capturedBody.enable_auto_chunking).toBeUndefined();
    expect(capturedBody.chunk_size).toBeUndefined();
    expect(vectors).toEqual([[7, 8]]);
  });

  it("returns an empty result without a request for empty input", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const vectors = await embedVoyageContextualized({
      client: buildClient(),
      inputs: [],
      inputType: "document",
    });
    expect(vectors).toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("throws on a malformed response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse({ data: [{ index: 0, data: [] }] })),
    );
    await expect(
      embedVoyageContextualized({
        client: buildClient(),
        inputs: ["alpha"],
        inputType: "document",
      }),
    ).rejects.toThrow(/malformed JSON response/);
  });
});
