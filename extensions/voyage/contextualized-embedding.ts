// Voyage contextualized-chunk embedding support (voyage-context-* models).
//
// Contextualized models embed each chunk with awareness of the surrounding
// document. We call the `contextualized_embed` API with a flat list[str] input,
// `enable_auto_chunking: true` and `chunk_size: 32000`. Because the chunk size
// equals the per-chunk context window, every document that fits the window
// yields exactly one chunk, so the response maps 1:1 back to the memory
// embedding provider contract (one vector per input text).
//
// See: https://docs.voyageai.com/docs/contextualized-chunk-embeddings
import {
  postJsonWithRetry,
  resolveEmbeddingEndpointUrl,
} from "openclaw/plugin-sdk/memory-core-host-engine-embeddings";
import type { VoyageEmbeddingClient } from "./embedding-provider.js";

/** Endpoint path for contextualized chunk embeddings. */
const CONTEXTUALIZED_EMBED_ENDPOINT = "contextualizedembeddings";

/**
 * Chunk size (tokens) used for contextualized embedding requests. Set to the
 * per-chunk context window so each in-limit document collapses to one chunk.
 */
export const VOYAGE_CONTEXTUALIZED_CHUNK_SIZE = 32000;

/** True when the model is a Voyage contextualized-chunk embedding model. */
export function isVoyageContextualizedModel(model: string): boolean {
  return model.startsWith("voyage-context");
}

type ContextualizedChunkEmbedding = {
  embedding?: unknown;
  index?: unknown;
};

type ContextualizedInputResult = {
  data?: ContextualizedChunkEmbedding[];
  index?: unknown;
};

type ContextualizedResponse = {
  data?: ContextualizedInputResult[];
};

function readContextualizedVectors(
  payload: ContextualizedResponse,
  expectedCount: number,
  errorPrefix: string,
): number[][] {
  const malformed = () => new Error(`${errorPrefix}: malformed JSON response`);
  const entries = payload.data;
  if (!Array.isArray(entries) || entries.length !== expectedCount) {
    throw malformed();
  }
  const vectors: number[][] = Array.from({ length: expectedCount });
  for (let position = 0; position < entries.length; position += 1) {
    const entry = entries[position];
    // Each input maps to a list of chunk embeddings; chunk_size == context
    // window means a single chunk per in-limit input, so take the first.
    const chunk = entry?.data?.[0];
    const embedding = chunk?.embedding;
    if (!Array.isArray(embedding) || embedding.length === 0) {
      throw malformed();
    }
    for (const coordinate of embedding) {
      if (typeof coordinate !== "number" || !Number.isFinite(coordinate)) {
        throw malformed();
      }
    }
    const index = typeof entry?.index === "number" ? entry.index : position;
    if (!Number.isInteger(index) || index < 0 || index >= expectedCount || vectors[index]) {
      throw malformed();
    }
    vectors[index] = embedding as number[];
  }
  return vectors;
}

/**
 * Embed inputs through the contextualized_embed API. Documents use server-side
 * auto-chunking with a full-window chunk size; queries are embedded directly
 * (auto-chunking requires `input_type: "document"`).
 */
export async function embedVoyageContextualized(params: {
  client: VoyageEmbeddingClient;
  inputs: string[];
  inputType: "query" | "document";
  signal?: AbortSignal;
}): Promise<number[][]> {
  if (params.inputs.length === 0) {
    return [];
  }
  const isDocument = params.inputType === "document";
  const payload = await postJsonWithRetry<ContextualizedResponse>({
    url: resolveEmbeddingEndpointUrl(params.client.baseUrl, CONTEXTUALIZED_EMBED_ENDPOINT),
    headers: params.client.headers,
    ssrfPolicy: params.client.ssrfPolicy,
    body: {
      model: params.client.model,
      // Flat list[str] input as required by contextualized auto-chunking.
      inputs: params.inputs,
      input_type: params.inputType,
      ...(isDocument
        ? { enable_auto_chunking: true, chunk_size: VOYAGE_CONTEXTUALIZED_CHUNK_SIZE }
        : {}),
    },
    errorPrefix: "voyage contextualized embeddings failed",
  });
  return readContextualizedVectors(
    payload,
    params.inputs.length,
    "voyage contextualized embeddings",
  );
}
