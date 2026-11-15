import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { voyageMemoryEmbeddingProviderAdapter } from "./memory-embedding-adapter.js";

export default definePluginEntry({
  id: "voyage",
  name: "Voyage AI by MongoDB Embeddings",
  description: "Voyage AI by MongoDB memory embedding provider plugin",
  register(api) {
    api.registerEmbeddingProvider(voyageMemoryEmbeddingProviderAdapter);
  },
});
