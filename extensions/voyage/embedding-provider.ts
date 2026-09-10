// Voyage provider module implements model/runtime integration.
import {
  buildRemoteBaseUrlPolicy,
  createRemoteEmbeddingProvider,
  normalizeEmbeddingModelWithPrefixes,
  resolveRemoteEmbeddingClient,
  type MemoryEmbeddingProvider,
  type MemoryEmbeddingProviderCreateOptions,
} from "openclaw/plugin-sdk/memory-core-host-engine-embeddings";
import type { SsrFPolicy } from "openclaw/plugin-sdk/ssrf-runtime";
import { normalizeOptionalString } from "openclaw/plugin-sdk/string-coerce-runtime";

export type VoyageEmbeddingClient = {
  baseUrl: string;
  headers: Record<string, string>;
  ssrfPolicy?: SsrFPolicy;
  model: string;
};

export const DEFAULT_VOYAGE_EMBEDDING_MODEL = "voyage-4-large";

// Voyage (by MongoDB) serves embeddings from api.voyageai.com. Keys issued
// through MongoDB Atlas use the "al-" prefix and must target ai.mongodb.com
// instead. This mirrors the official voyageai-python client, which selects the
// base URL from the API key prefix in get_default_base_url:
// https://github.com/voyage-ai/voyageai-python/blob/9aca465efd0011c478031f6d584b70a3a4393a7c/voyageai/util.py#L99
export const VOYAGE_BASE_URL = "https://api.voyageai.com/v1";
export const VOYAGE_MONGODB_BASE_URL = "https://ai.mongodb.com/v1";
const MONGODB_API_KEY_PREFIX = "al-";

// Max input tokens per model, from https://docs.voyageai.com/docs/embeddings.
// Current models are the voyage-4 series plus the domain-specific code/finance/
// law models; legacy voyage-3 rows stay so existing memory indexes keep their
// token budget.
const VOYAGE_MAX_INPUT_TOKENS: Record<string, number> = {
  "voyage-4-large": 32000,
  "voyage-4": 32000,
  "voyage-4-lite": 32000,
  "voyage-code-4": 32000,
  "voyage-finance-2": 32000,
  "voyage-law-2": 16000,
  // Legacy models retained for existing indexes.
  "voyage-code-2": 16000,
  "voyage-3": 32000,
  "voyage-3-lite": 16000,
  "voyage-code-3": 32000,
};

function normalizeVoyageModel(model: string): string {
  return normalizeEmbeddingModelWithPrefixes({
    model,
    defaultModel: DEFAULT_VOYAGE_EMBEDDING_MODEL,
    prefixes: ["voyage/"],
  });
}

/**
 * Resolve the Voyage default base URL from the API key prefix, matching the
 * official voyageai-python `get_default_base_url`. MongoDB Atlas keys ("al-")
 * route to ai.mongodb.com; every other key stays on api.voyageai.com.
 */
export function getDefaultVoyageBaseUrl(apiKey: string | undefined): string {
  return apiKey?.startsWith(MONGODB_API_KEY_PREFIX) ? VOYAGE_MONGODB_BASE_URL : VOYAGE_BASE_URL;
}

/** Read the bearer token from resolved request headers, ignoring header case. */
function extractBearerToken(headers: Record<string, string>): string | undefined {
  for (const [name, value] of Object.entries(headers)) {
    if (name.toLowerCase() !== "authorization") {
      continue;
    }
    const match = /^Bearer\s+(.+)$/iu.exec(value.trim());
    return match?.[1]?.trim();
  }
  return undefined;
}

/** Whether the user pinned an explicit Voyage base URL via config or memory remote. */
function hasExplicitVoyageBaseUrl(options: MemoryEmbeddingProviderCreateOptions): boolean {
  const remoteBaseUrl = normalizeOptionalString(options.remote?.baseUrl);
  const providerBaseUrl = normalizeOptionalString(
    options.config.models?.providers?.voyage?.baseUrl,
  );
  return Boolean(remoteBaseUrl || providerBaseUrl);
}

export async function createVoyageEmbeddingProvider(
  options: MemoryEmbeddingProviderCreateOptions,
): Promise<{ provider: MemoryEmbeddingProvider; client: VoyageEmbeddingClient }> {
  const client = await resolveRemoteEmbeddingClient({
    provider: "voyage",
    options,
    defaultBaseUrl: VOYAGE_BASE_URL,
    normalizeModel: normalizeVoyageModel,
  });
  // Redirect MongoDB Atlas keys to ai.mongodb.com, but only when the caller did
  // not pin an explicit base URL. An override always wins, exactly as the
  // official client treats an explicit base_url. The SSRF policy is derived from
  // the base URL host, so it must be rebuilt for the redirected destination.
  if (!hasExplicitVoyageBaseUrl(options)) {
    const redirected = getDefaultVoyageBaseUrl(extractBearerToken(client.headers));
    if (redirected !== client.baseUrl) {
      client.baseUrl = redirected;
      client.ssrfPolicy = buildRemoteBaseUrlPolicy(redirected);
    }
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
