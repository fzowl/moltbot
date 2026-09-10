// Voyage provider tests cover MongoDB base-URL redirection and model budgets.
import { describe, expect, it } from "vitest";
import {
  createVoyageEmbeddingProvider,
  getDefaultVoyageBaseUrl,
  VOYAGE_BASE_URL,
  VOYAGE_MONGODB_BASE_URL,
} from "./embedding-provider.js";

describe("getDefaultVoyageBaseUrl", () => {
  it("routes MongoDB Atlas keys to ai.mongodb.com", () => {
    expect(getDefaultVoyageBaseUrl("al-fixture-key")).toBe(VOYAGE_MONGODB_BASE_URL);
  });

  it("keeps standard Voyage keys and missing keys on api.voyageai.com", () => {
    expect(getDefaultVoyageBaseUrl("pa-fixture-key")).toBe(VOYAGE_BASE_URL);
    expect(getDefaultVoyageBaseUrl(undefined)).toBe(VOYAGE_BASE_URL);
  });
});

describe("createVoyageEmbeddingProvider base URL redirection", () => {
  it("redirects al- keys to the MongoDB endpoint and rebuilds the SSRF policy", async () => {
    const { client } = await createVoyageEmbeddingProvider({
      config: {},
      provider: "voyage",
      model: "voyage-4-large",
      fallback: "none",
      remote: { apiKey: "al-fixture-key" },
    });
    expect(client.baseUrl).toBe(VOYAGE_MONGODB_BASE_URL);
    expect(client.ssrfPolicy).toBeDefined();
  });

  it("keeps standard keys on api.voyageai.com", async () => {
    const { client } = await createVoyageEmbeddingProvider({
      config: {},
      provider: "voyage",
      model: "voyage-4-large",
      fallback: "none",
      remote: { apiKey: "pa-fixture-key" },
    });
    expect(client.baseUrl).toBe(VOYAGE_BASE_URL);
  });

  it("honors an explicit base URL override even for al- keys", async () => {
    const { client } = await createVoyageEmbeddingProvider({
      config: {},
      provider: "voyage",
      model: "voyage-4-large",
      fallback: "none",
      remote: { apiKey: "al-fixture-key", baseUrl: "https://voyage.example/v1" },
    });
    expect(client.baseUrl).toBe("https://voyage.example/v1");
  });

  it("assigns current models their documented max input token budget", async () => {
    const large = await createVoyageEmbeddingProvider({
      config: {},
      provider: "voyage",
      model: "voyage-4-large",
      fallback: "none",
      remote: { apiKey: "pa-fixture-key" },
    });
    expect(large.provider.maxInputTokens).toBe(32000);

    const law = await createVoyageEmbeddingProvider({
      config: {},
      provider: "voyage",
      model: "voyage-law-2",
      fallback: "none",
      remote: { apiKey: "pa-fixture-key" },
    });
    expect(law.provider.maxInputTokens).toBe(16000);
  });
});
