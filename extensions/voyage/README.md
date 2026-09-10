# OpenClaw Voyage Provider

Official OpenClaw memory embedding provider plugin for VoyageAI by MongoDB.

Install from OpenClaw:

```bash
openclaw plugins install @openclaw/voyage-provider
openclaw gateway restart
```

Set `VOYAGE_API_KEY`, then configure memory search with `provider: "voyage"`.
See <https://docs.openclaw.ai/reference/memory-config> for setup and
configuration.

Keys issued through MongoDB Atlas (prefixed `al-`) are routed to
`https://ai.mongodb.com/v1` automatically; all other keys use
`https://api.voyageai.com/v1`. Set `models.providers.voyage.baseUrl` (or
`memory.search.remote.baseUrl`) to override the destination.
