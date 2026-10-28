# Cloudflare Jev and a tags-only handoff

This fork can call Cloudflare's native `typesafe/jev` model through Workers AI.
It also supports polling a dedicated queue tag, so Paperless-GPT can finish OCR
and metadata before Jev handles semantic tags.

## Credentials and transport

Set these environment variables on the Jev container:

```yaml
environment:
  CLOUDFLARE_ACCOUNT_ID: your-account-id
  CLOUDFLARE_API_TOKEN_FILE: /run/secrets/cloudflare-token
volumes:
  - ./secrets/cloudflare-token:/run/secrets/cloudflare-token:ro
```

Create the token file through your secret manager, outside the repository. Grant
only the inference permissions needed for the selected Cloudflare account. The
container runs as UID 10001; allow that user to read the file, for example with
owner 10001 and mode 0400. Do not put the token value in Compose, command
arguments, logs or Git. Keep `/data` and its vault key private too.

Keep the classifier set to TypeSafe in the web UI. With the Cloudflare account
configured, this backend uses Cloudflare and the model `typesafe/jev`; it does
not require a direct TypeSafe API key. An explicitly selected Ollama backend
continues to use its configured local endpoint.

The adapter posts to `/client/v4/accounts/{account}/ai/run` with the model and
the original Jev `state` and `questions` under `input`. It accepts direct and
completed wrapped answers. Failed or incomplete results and invalid tag
probabilities fail the job instead of marking the document complete. Missing
Cloudflare credentials do not trigger a fallback to the direct TypeSafe service.

Document text and the classification criteria are sent to Cloudflare. Review
that data flow before enabling this backend for private documents. To avoid
sending archive examples, set both `examples` and `similar_docs` to zero.

## Sequence Jev after Paperless-GPT

1. Create a dedicated tag such as `jev-pending` in Paperless. Do not assign it
   to your existing archive unless you intend to process that archive.
2. In Jev, configure tags only, turn title generation off, disable verification
   of existing tags, and preserve inbox tags. Start in dry-run mode with
   polling disabled. Review representative document results first.
3. Set the advanced Jev settings `poll_tag` and `system_tags`. These are stored
   settings, not web-UI controls. For example, run this inside the Jev container
   against its existing data directory:

   ```python
   import os
   from pathlib import Path
   from paperless_jev.config import Config
   from paperless_jev.db import Database
   from paperless_jev.vault import Vault

   directory = Path(os.environ.get("PJ_DATA_DIR", "/data"))
   config = Config(Database(directory / "paperless-jev.db"), Vault(directory))
   settings = config.all()
   config.update({
       "poll_tag": "jev-pending",
       "system_tags": sorted(set(settings.get("system_tags", [])) | {
           "jev-pending", "paperless-gpt-auto", "paperless-gpt-ocr-auto",
           "paperless-gpt-complete", "ocr-complete",
       }),
   })
   ```

   Use the actual processing-tag names from your installation. `system_tags`
   excludes those tags from semantic classification and cleanup suggestions.
4. When ready to switch, set Paperless-GPT's `AUTO_GENERATE_TAGS=false` and
   `AUTO_TAG_COMPLETE=jev-pending`. Keep its other metadata and OCR settings.
5. Enable automatic mode and a polling interval in Jev. A conservative starting
   point is an automatic tag threshold of 0.90, review threshold of 0.70, and
   `tags_force_review=true` so uncertain tags enter review.

When `poll_tag` is set, Jev polls only documents with that tag. A missing tag
produces an empty queue, never a fallback to the inbox. Jev removes the queue
tag after successfully writing either its completed or review result. Errors
and dry runs keep the queue tag. As upstream does, errored jobs require an
explicit retry; retaining the tag does not imply an automatic retry loop.
Without `poll_tag`, upstream inbox polling is unchanged.

Keep the administration interface authenticated and private, for example with
a loopback-only published port. A polling handoff does not require a public
webhook or a new consume mount.

Verify that your existing GPT OCR pipeline terminates before enabling the
handoff. Replacement PDFs that regain the OCR inbox tag without a completion
marker can loop before metadata processing reaches `AUTO_TAG_COMPLETE`.
This fork does not change Paperless-GPT's OCR behavior.

## Tests and builds

Install the development dependencies and run `pytest -q` as upstream does. The
Cloudflare/handoff regression tests use synthetic text and mocked HTTP; they
require neither credentials nor a Paperless instance.

Build a local image using the repository's Dockerfile. The fork's workflow
runs tests but restricts automatic image publication to the upstream repository.
Publishing your own image requires a separate workflow change.
