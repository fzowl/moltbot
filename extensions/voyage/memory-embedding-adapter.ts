import {
  mapBatchEmbeddingsByIndex,
  sanitizeEmbeddingCacheHeaders,
} from "openclaw/plugin-sdk/embedding-provider-adapter";
import {
  createRemoteEmbeddingProvider,
  normalizeEmbeddingModelWithPrefixes,
  resolveEmbeddingProviderSsrFPolicy,
  resolveRemoteEmbeddingClient,
  type MemoryEmbeddingProviderAdapter,
  type MemoryEmbeddingProviderCreateOptions,
} from "openclaw/plugin-sdk/memory-core-host-engine-embeddings";
import { runVoyageEmbeddingBatches } from "./embedding-batch.js";

const DEFAULT_VOYAGE_EMBEDDING_MODEL = "voyage-4-large";
const DEFAULT_VOYAGE_BASE_URL = "https://api.voyageai.com/v1";
// Keys issued through MongoDB Atlas start with "al-" and are served from a different host.
// Mirrors get_default_base_url() in the official voyageai-python client.
const MONGODB_VOYAGE_BASE_URL = "https://ai.mongodb.com/v1";
const MONGODB_API_KEY_PREFIX = "al-";
// Context window per input, from https://docs.voyageai.com/docs/embeddings.
const VOYAGE_MAX_INPUT_TOKENS: Record<string, number> = {
  "voyage-4-large": 32000,
  "voyage-4": 32000,
  "voyage-4-lite": 32000,
  "voyage-code-4": 32000,
  "voyage-3-large": 32000,
  "voyage-3.5": 32000,
  "voyage-3.5-lite": 32000,
  "voyage-code-3": 32000,
  "voyage-finance-2": 32000,
  "voyage-law-2": 16000,
  "voyage-3": 32000,
  "voyage-3-lite": 32000,
};
// A /v1/embeddings request accepts at most 1,000 inputs.
const VOYAGE_MAX_INPUTS_PER_REQUEST = 1000;

export const voyageMemoryEmbeddingProviderAdapter: MemoryEmbeddingProviderAdapter = {
  id: "voyage",
  defaultModel: DEFAULT_VOYAGE_EMBEDDING_MODEL,
  transport: "remote",
  authProviderId: "voyage",
  autoSelectPriority: 40,
  create: async (options) => {
    const preparedOptions: MemoryEmbeddingProviderCreateOptions = {
      ...options,
      provider: "voyage",
      fallback: "none",
    };
    // Matryoshka models accept 2048, 1024, 512 or 256; Voyage rejects other sizes per model.
    const outputDimension = options.dimensions;
    const client = await resolveRemoteEmbeddingClient({
      provider: "voyage",
      options: preparedOptions,
      defaultBaseUrl: DEFAULT_VOYAGE_BASE_URL,
      normalizeModel: (model) =>
        normalizeEmbeddingModelWithPrefixes({
          model,
          defaultModel: DEFAULT_VOYAGE_EMBEDDING_MODEL,
          prefixes: ["voyage/"],
        }),
    });
    // The key picks the default host; an explicit base URL always wins, as in the official client.
    const providerConfig = options.config.models?.providers?.voyage;
    const hasExplicitBaseUrl = Boolean(
      options.remote?.baseUrl?.trim() || providerConfig?.baseUrl?.trim(),
    );
    const authorization = Object.entries(client.headers).find(
      ([name]) => name.toLowerCase() === "authorization",
    )?.[1];
    if (
      !hasExplicitBaseUrl &&
      authorization?.replace(/^Bearer\s+/iu, "").startsWith(MONGODB_API_KEY_PREFIX)
    ) {
      client.baseUrl = MONGODB_VOYAGE_BASE_URL;
      client.ssrfPolicy = resolveEmbeddingProviderSsrFPolicy({
        provider: "voyage",
        api: providerConfig?.api ?? "openai-completions",
        baseUrl: MONGODB_VOYAGE_BASE_URL,
        allowPrivateNetwork: providerConfig?.request?.allowPrivateNetwork,
      });
    }
    const provider = createRemoteEmbeddingProvider({
      id: "voyage",
      client,
      errorPrefix: "voyage embeddings failed",
      maxInputsPerRequest: VOYAGE_MAX_INPUTS_PER_REQUEST,
      buildRequestFields: (kind) => ({
        input_type: kind,
        ...(outputDimension !== undefined ? { output_dimension: outputDimension } : {}),
      }),
    });
    provider.maxInputTokens = VOYAGE_MAX_INPUT_TOKENS[client.model];
    return {
      provider,
      runtime: {
        id: "voyage",
        cacheKeyData: {
          provider: "voyage",
          baseUrl: client.baseUrl,
          model: client.model,
          ...(outputDimension !== undefined ? { outputDimension } : {}),
          headers: sanitizeEmbeddingCacheHeaders(client.headers, ["authorization"]),
        },
        batchEmbed: async (batch) => {
          const byCustomId = await runVoyageEmbeddingBatches({
            ...batch,
            client,
            outputDimension,
            requests: batch.chunks.map((chunk, index) => ({
              custom_id: String(index),
              body: { input: chunk.text },
            })),
          });
          return mapBatchEmbeddingsByIndex(byCustomId, batch.chunks.length);
        },
      },
    };
  },
};
