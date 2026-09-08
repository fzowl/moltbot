// Voyage plugin entrypoint registers its OpenClaw integration.
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { voyageMemoryEmbeddingProviderAdapter } from "./memory-embedding-adapter.js";

export default definePluginEntry({
  id: "voyage",
  name: "VoyageAI by MongoDB Embeddings",
  description: "VoyageAI by MongoDB memory embedding provider plugin",
  register(api) {
    api.registerEmbeddingProvider(voyageMemoryEmbeddingProviderAdapter);
  },
});
