// Voyage provider module implements model/runtime integration.
import {
  createRemoteEmbeddingProvider,
  normalizeEmbeddingModelWithPrefixes,
  resolveRemoteEmbeddingClient,
  type MemoryEmbeddingProvider,
  type MemoryEmbeddingProviderCreateOptions,
} from "openclaw/plugin-sdk/memory-core-host-engine-embeddings";
import type { SsrFPolicy } from "openclaw/plugin-sdk/ssrf-runtime";
import {
  embedVoyageContextualized,
  isVoyageContextualizedModel,
  VOYAGE_CONTEXTUALIZED_CHUNK_SIZE,
} from "./contextualized-embedding.js";

export type VoyageEmbeddingClient = {
  baseUrl: string;
  headers: Record<string, string>;
  ssrfPolicy?: SsrFPolicy;
  model: string;
};

export const DEFAULT_VOYAGE_EMBEDDING_MODEL = "voyage-4-large";
const DEFAULT_VOYAGE_BASE_URL = "https://api.voyageai.com/v1";
// Max input tokens per Voyage AI (VoyageAI by MongoDB) embedding model.
// Current generation first, then legacy models kept for API back-compat.
// See: https://docs.voyageai.com/docs/embeddings
const VOYAGE_MAX_INPUT_TOKENS: Record<string, number> = {
  // Current general-purpose, domain-specific, and open-weight models.
  "voyage-4-large": 32000,
  "voyage-4": 32000,
  "voyage-4-lite": 32000,
  "voyage-4-nano": 32000,
  "voyage-code-4": 32000,
  "voyage-finance-2": 32000,
  "voyage-law-2": 16000,
  // Contextualized-chunk embedding model (per-chunk context window).
  "voyage-context-4": 32000,
  // Legacy models still accessible through the API.
  "voyage-3-large": 32000,
  "voyage-3.5": 32000,
  "voyage-3.5-lite": 32000,
  "voyage-3": 32000,
  "voyage-3-lite": 16000,
  "voyage-code-3": 32000,
  "voyage-context-3": 32000,
};

function normalizeVoyageModel(model: string): string {
  return normalizeEmbeddingModelWithPrefixes({
    model,
    defaultModel: DEFAULT_VOYAGE_EMBEDDING_MODEL,
    prefixes: ["voyage/"],
  });
}

export async function createVoyageEmbeddingProvider(
  options: MemoryEmbeddingProviderCreateOptions,
): Promise<{ provider: MemoryEmbeddingProvider; client: VoyageEmbeddingClient }> {
  const client = await resolveRemoteEmbeddingClient({
    provider: "voyage",
    options,
    defaultBaseUrl: DEFAULT_VOYAGE_BASE_URL,
    normalizeModel: normalizeVoyageModel,
  });
  // Contextualized-chunk models use a separate endpoint and input shape.
  if (isVoyageContextualizedModel(client.model)) {
    return { provider: createVoyageContextualizedEmbeddingProvider(client), client };
  }
  const provider = createRemoteEmbeddingProvider({
    id: "voyage",
    client,
    errorPrefix: "voyage embeddings failed",
    buildRequestFields: (kind) => ({ input_type: kind }),
  });
  provider.maxInputTokens = VOYAGE_MAX_INPUT_TOKENS[client.model];
  return { provider, client };
}

function toEmbeddingText(input: string | { text: string }): string {
  return typeof input === "string" ? input : input.text;
}

/** Build a provider that embeds through the contextualized_embed API. */
function createVoyageContextualizedEmbeddingProvider(
  client: VoyageEmbeddingClient,
): MemoryEmbeddingProvider {
  const resolveKind = (inputType?: string): "query" | "document" =>
    inputType === "query" ? "query" : "document";
  return {
    id: "voyage",
    model: client.model,
    maxInputTokens: VOYAGE_MAX_INPUT_TOKENS[client.model] ?? VOYAGE_CONTEXTUALIZED_CHUNK_SIZE,
    embed: async (input, options) => {
      const [vec] = await embedVoyageContextualized({
        client,
        inputs: [toEmbeddingText(input)],
        inputType: resolveKind(options?.inputType),
        signal: options?.signal,
      });
      return vec ?? [];
    },
    embedBatch: async (inputs, options) =>
      await embedVoyageContextualized({
        client,
        inputs: inputs.map(toEmbeddingText),
        inputType: resolveKind(options?.inputType),
        signal: options?.signal,
      }),
  };
}
