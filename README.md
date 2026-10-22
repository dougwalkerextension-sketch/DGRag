# DGRag

A local research dashboard and Poe literary simulator using three configured sources:

- Volume 1, ebook 2147: https://www.gutenberg.org/cache/epub/2147/pg2147.txt
- Volume 2, ebook 2148: https://www.gutenberg.org/cache/epub/2148/pg2148.txt
- [The Philosophy of Composition (1846)](https://www.eapoe.org/works/essays/philcomp.htm), Text-02, hosted by the Edgar Allan Poe Society of Baltimore (Graham's Magazine, April 1846).

## Run

This workspace includes a portable Python runtime and installed dependencies. Run:

```powershell
.\start.ps1
```

Open http://127.0.0.1:8000. The application binds to localhost only. **Load Poe texts** downloads or reads cached copies of all three sources, validates Gutenberg markers and ebook IDs, removes Gutenberg wrappers, and extracts the essay's text body without site navigation, page labels, scripts, or editorial notes. Missing or incomplete essay bodies are rejected. This requires no API keys. Raw text/HTML, cleaned bodies, prepared chunks, run history, and activity persist under ignored `data/`.

**Index in Pinecone** embeds and stores the text after credentials are configured in `.env`. Restart after changing configuration. Set `PINECONE_API_KEY` and `PINECONE_INDEX_HOST` for indexing. Use a Pinecone dense vector index with cosine similarity and **384 dimensions** for the default local `sentence-transformers/all-MiniLM-L6-v2` model. Local answer generation needs no API key. The optional OpenAI comparison output uses OPENAI_API_KEY. `PINECONE_INDEX_HOST` is the index hostname ending in `.pinecone.io`. Credentials are never included in dashboard responses.

For a fresh checkout with Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe app.py --download-model
.\.venv\Scripts\python.exe app.py
```

## Ask Poe

Select **Ask Poe** to converse with a literary simulation of Edgar Allan Poe about characters, feelings, recurring themes, or imagery such as ravens and clocks. Suggested questions help begin a conversation. Follow-ups include recent dialogue in retrieval, using the same local Hugging Face embedding model and Pinecone index as the library. Each prompt creates three horizontally arranged output text boxes: the current local Hugging Face Qwen model on the left, all-MiniLM-L6-v2 retrieved passages in the center, and OpenAI on the right. MiniLM is an embedding model: its output is source passages, not a second generative Poe reply. Qwen and optional OpenAI generate independently in a Poe-inspired voice using the same work-aware retrieval and composition-essay evidence.

After shared MiniLM/Pinecone retrieval, Qwen and OpenAI run concurrently. The streaming `/api/poe` response sends each completed output separately, so a slow or failed model does not hide other outputs. Source passages remain expandable below each row. Saved browser history retains all three outputs, and each generative model receives only its own successful prior replies along with recent user prompts. Old saved conversations remain readable. Smaller screens scroll horizontally to preserve the left/center/right order.

Set `OPENAI_API_KEY` and optionally `OPENAI_CHAT_MODEL` (default `gpt-4.1-mini`) in `.env` to enable the right-hand output. The key is read server-side on each prompt and never sent to the browser or exported. When absent or blank, that box says exactly **not connected** and makes no OpenAI request. A configured key enables one Responses API request per Ask Poe prompt, sending the prompt, that model's recent history, and retrieved passages with `store: false`. OpenAI authentication, network, or quota errors appear only in its box; configured does not imply the key has been validated. Research mode remains local Qwen only. [Official API guide](https://developers.openai.com/api/docs/guides/text).

The Gutenberg files are anthologies, not continuous narratives. Their contents entries are resolved to body headings, yielding **30 separate literary works** (8 in Volume 1 and 22 in Volume 2). Chunks overlap only within the same work. Each chunk has a stable work ID, work title, parent volume, work type, and offsets in both the work and volume. Contents pages, introductions, and the second volume's endnotes are separate editorial sections; these are excluded from default Ask Poe story retrieval. Notes are not silently attached to a story.

Naming a tale scopes retrieval to that work; follow-ups inherit the most recent user-named work unless the current question names another. Naming multiple tales allocates retrieval slots to each for supported comparisons (up to 6 in Ask Poe or 8 in research). Assistant text cannot choose the work for a follow-up. Broad thematic questions still search across works. Replies must attribute characters, events, speakers, and quotations to the specific work in each excerpt and must not combine different narrators into one. Story citations show both the work and the parent volume. The source collection lists the distinct works.

The simulator uses **The Philosophy of Composition** as its interpretive frame: intended effect, planning the ending, unity of impression, and deliberate choices of tone, beauty, melancholy, sound, refrain, contrast, and suggestiveness. This is Poe's published account, not access to private thoughts or proof that every work followed the same method. Applications to other works must be labeled interpretation and supported by those works' passages. Each Ask Poe turn reserves up to two retrieved essay chunks and six anthology chunks, with essay evidence first to preserve it during context trimming. Essay claims use the same numbered citations and inspectable source links as the tales. Research mode searches all three sources for eight chunks without reserved slots. No essay text is injected as trusted instructions.

**Upgrading from any earlier corpus version (including the volume-wide essay version):** restart the server, select **Load Poe texts**, then **Index in Pinecone**. Both steps must complete with all three sources. Indexes without the current work boundaries and metadata are rejected with a reindex message; preparing documents alone does not activate a new corpus. Partial JSON imports remain available for research, but Ask Poe requires an indexed essay and refuses to generate if its passages are unavailable. Existing Pinecone namespaces are preserved.

Each reply has expandable supporting passages and numbered references. Conversation history is retained in this browser (up to 20 exchanges), with the last six exchanges sent for context. **New conversation** resets the visible browser conversation. Comparison exchanges and per-model status also remain in the application's exported answer log. Conversations require a completed Pinecone ingestion. The left column requires the cached local answer model; a missing local generator does not block MiniLM or configured OpenAI outputs. The tab supports keyboard navigation and can be opened directly at http://127.0.0.1:8000/#ask-poe.

## Local embedding model

Download the public model once before indexing:

```powershell
.\.runtime\python.exe app.py --download-model
```

This stores model weights and tokenizer files in `data/huggingface/`. Runtime loads use `local_files_only=True`, CPU inference, and `trust_remote_code=False`; embedding calls need neither internet nor a Hugging Face API token. Initial model acquisition requires internet. MiniLM embeddings and Qwen inference run locally. Pinecone stores and retrieves vectors; the optional OpenAI column uses remote generation when its key is set.

`EMBEDDING_MODEL` selects the model or a local Sentence Transformers folder; `EMBEDDING_REVISION` pins the Hub revision. The default MiniLM model is pinned in `.env.example`. For a custom model, update or clear that revision and match the Pinecone dimension to its output. The default dimension changed from 1536 to **384**; create or select a matching index and reindex. No existing index is deleted or resized.

Model source: [Hugging Face model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

## Local answer generation

The default answer model is **Qwen/Qwen2.5-0.5B-Instruct**, a separate causal language model from the MiniLM embedding model. It runs on CPU using Transformers, with a pinned revision, local-only weight loading, and no remote code or inference fallback. Download its approximately 1 GB of weights once:

```powershell
.\.runtime\python.exe app.py --download-generator
```

Set `GENERATION_MODEL`, `GENERATION_REVISION`, and `GENERATION_MAX_NEW_TOKENS` in `.env` to configure it, then restart. Local generation needs several GB of free RAM and can be slower than a hosted model. This small default model prioritizes local CPU use; interpretive quality and citation reliability may be limited. A larger compatible instruction model can be selected if your hardware supports it. The prompt retains current evidence and trims older conversation first when the token budget is exceeded. Pinecone indexing does not need to be repeated when changing only the answer model.

No OpenAI key is required for local inference. It is used only for the optional OpenAI comparison column. [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

## Pipeline and results

1. **Fetch:** only the three URLs above are configured. The app validates UTF-8 text, Gutenberg START/END markers and ebook IDs, and the essay's HTML body boundaries, identifying text, and ending. Existing valid downloads are reused. To download new copies, run `.\.runtime\python.exe app.py --prepare --refresh` while the server is stopped. Raw Gutenberg files retain their license; the raw essay HTML retains its source attribution.
2. **Prepare:** anthology contents are matched to distinct work headings, including the Hans Pfaall/Pfaal and William Wilso/Wilson variants in these editions. Missing or ambiguous work headings stop ingestion. Each work or editorial section is split into 2,100-character windows with 300-character overlap; no window crosses a work boundary. Metadata includes the source title, author, ebook/source ID (`philcomp` for the essay), source type, URL, body character offsets, capture time, and normalized document checksum. Body offsets refer to cleaned text, not printed pages. Editorial material within anthology bodies remains available and is distinguished in the answer prompt; the essay site's separate editorial notes are excluded.
3. **Embed:** the Hugging Face model runs locally on CPU through Sentence Transformers. Each input is split into windows within the model context limit; normalized window vectors are averaged and normalized again. This embeds the full passage without silent truncation. Events expose the model, pinned revision, pooling settings, batch sizes, and 384-dimensional output. There are no OpenAI embedding calls.
4. **Store:** Pinecone upsert receipts are checked. Each run uses a fresh `dgrag-poe-…` namespace; it becomes active only when all batches succeed. Old namespaces remain for audit and manual cleanup. Failed writes preserve the previous active namespace. Pinecone can take a moment to make writes queryable.
5. **Retrieve:** questions use the same local model, revision, normalization, and pooling settings as ingestion and retrieve eight chunks from the active namespace. Old OpenAI indexes and changed model settings are rejected; reindex after switching.
6. **Answer:** the local Hugging Face Qwen model generates an answer constrained to the retrieved text, with numbered citations. Ask Poe additionally uses the essay's composition framework. The dashboard shows similarity scores and exact passages. Coverage is limited to Volumes 1 and 2 and the composition essay, not all of Poe's works.

The dashboard includes a searchable source collection, every document chunk, event details, question history, and **Export results** for full JSON. Provider secrets and full embedding arrays are excluded. State uses `data/poe-state.json`; the active corpus version and indexed source URLs are recorded only after successful ingestion, so older two-volume indexes require an explicit upgrade.

Optional JSON imports accept one to three configured sources, each with `url` and `text` containing the **full raw Gutenberg file including its markers and ebook header**, or the **full original HTML** for the configured essay URL. Other URLs and duplicates are rejected. Import all three to enable the complete simulator. Example shape:

```json
[{"url":"https://www.gutenberg.org/cache/epub/2147/pg2147.txt","text":"Full raw file contents here"}]
```

## Pinecone connection troubleshooting

Run the app from a network-enabled terminal using `.\.runtime\python.exe app.py`. A network-restricted agent sandbox can produce Windows socket error 10013 even with valid credentials.

The index must accept locally generated dense vectors: 384 dimensions, cosine similarity. A document-schema index configured with Pinecone hosted embeddings is incompatible with this application's vector API. `scripts/configure_pinecone.py` creates or selects `dgrag-poe-local`, preserves existing indexes, and updates the host in `.env`. Restart after changing the host. Network errors and incompatible-index errors now report specific recovery steps without exposing provider response bodies.

## Validate

```powershell
.\.runtime\python.exe -m unittest discover -s tests -v
.\.runtime\python.exe app.py --prepare
.\.runtime\python.exe scripts/check_local_embeddings.py
.\.runtime\python.exe scripts/check_local_generation.py
```

The unit suite uses mocked model execution and provider responses, including essay extraction and malformed-page rejection, three-source caching, reserved essay retrieval, framing instructions, missing essay evidence, and old-index migration, work boundaries, heading aliases, title-scoped follow-ups, balanced comparison retrieval, concurrent model execution, missing-key behavior, Responses API payloads, and isolated provider failures. Separate embedding and generation checks run the actual cached models on Poe passages with network connections blocked. Receipts are saved to `data/local-embedding-check.json` and `data/local-generation-check.json`. Pinecone access requires its credentials; optional OpenAI generation requires its own key. Local generation requires downloaded model weights. The application is a single-user local tool. WORK_SUMMARY.md retains the work record.

## GitHub publication

Source repository: https://github.com/dougwalkerextension-sketch/DGRag. `scripts/package_repository.py` creates a source archive and manifest, checks included files against configured secrets, and excludes `.env`, local data, downloaded models, and runtime binaries. New checkouts must fetch all three sources, download both models, and index before using Ask Poe. Publishing source does not host the application.
