# OpenClaw Voyage AI by MongoDB Provider

Official OpenClaw memory embedding provider plugin for Voyage AI by MongoDB.

Install from OpenClaw:

```bash
openclaw plugins install @openclaw/voyage-provider
openclaw gateway restart
```

Set `VOYAGE_API_KEY`, then configure memory search with `provider: "voyage"`.
See <https://docs.openclaw.ai/reference/memory-config> for setup and
configuration.

Keys that start with `al-` (issued through MongoDB Atlas) use
`https://ai.mongodb.com/v1`; all other keys use `https://api.voyageai.com/v1`.
An explicit `baseUrl` always wins.
