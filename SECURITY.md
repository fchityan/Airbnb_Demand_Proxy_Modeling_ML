# Security Policy

Do not commit API keys, cloud credentials, raw customer/search/contact data, signed artifact URLs, or production secrets.

Production should use TLS ingress, a secret manager, `REQUIRE_API_KEY=true`, a pinned `MODEL_SHA256`, read-only model mounts, non-root containers, network policy, gateway rate/request-size limits, and organization-standard dependency/container/secret scanning.

Rotate exposed credentials first, then remove them from Git history.
