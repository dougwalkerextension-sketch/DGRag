# DGRag

A local research dashboard for **The Works of Edgar Allan Poe, Volumes 1 and 2**, using only these Project Gutenberg texts:

- Volume 1, ebook 2147: https://www.gutenberg.org/cache/epub/2147/pg2147.txt
- Volume 2, ebook 2148: https://www.gutenberg.org/cache/epub/2148/pg2148.txt

## Run

This workspace includes a portable Python runtime and installed dependencies. Run:

```powershell
.\start.ps1
```

Open http://127.0.0.1:8000. The application binds to localhost only. **Load Poe texts** downloads or reads cached copies of both volumes, validates Gutenberg markers and ebook IDs, removes the header and license footer, and prepares chunks. This requires no API keys. Raw text, cleaned bodies, prepared chunks, run history, and activity persist under ignored `data/`.

**Index in Pinecone** embeds and stores the text after credentials are configured in `.env`. Restart after changing configuration. Set `PINECONE_API_KEY` and `PINECONE_INDEX_HOST` for indexing. Use a Pinecone dense vector index with cosine similarity and **384 dimensions** for the default local `sentence-transformers/all-MiniLM-L6-v2` model. Answer generation runs locally and needs no API key. `PINECONE_INDEX_HOST` is the index hostname ending in `.pinecone.io`. Credentials are never included in dashboard responses.

For a fresh checkout with Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe app.py --download-model
.\.venv\Scripts\python.exe app.py
```

## Ask Poe

Select **Ask Poe** to converse with a literary simulation of Edgar Allan Poe about characters, feelings, recurring themes, or imagery such as ravens and clocks. Suggested questions help begin a conversation. Follow-ups include recent dialogue in retrieval, using the same local Hugging Face embedding model and Pinecone index as the library. The local Hugging Face Qwen model generates the replies in a Poe-inspired voice, constrained to the retrieved passages; missing evidence is acknowledged rather than invented.

Each reply has expandable supporting passages and numbered references. Conversation history is retained in this browser (up to 20 exchanges), with the last six exchanges sent for context. **New conversation** resets the visible browser conversation. Successful exchanges also remain in the application's exported answer log. Conversations require a completed Pinecone ingestion and the cached local answer model. The tab supports keyboard navigation and can be opened directly at http://127.0.0.1:8000/#ask-poe.

## Local embedding model

Download the public model once before indexing:

```powershell
.\.runtime\python.exe app.py --download-model
```

This stores model weights and tokenizer files in `data/huggingface/`. Runtime loads use `local_files_only=True`, CPU inference, and `trust_remote_code=False`; embedding calls need neither internet nor a Hugging Face API token. Initial model acquisition requires internet. Only Pinecone storage and retrieval use a remote service; both embedding and answer inference run locally.

`EMBEDDING_MODEL` selects the model or a local Sentence Transformers folder; `EMBEDDING_REVISION` pins the Hub revision. The default MiniLM model is pinned in `.env.example`. For a custom model, update or clear that revision and match the Pinecone dimension to its output. The default dimension changed from 1536 to **384**; create or select a matching index and reindex. No existing index is deleted or resized.

Model source: [Hugging Face model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

## Local answer generation

The default answer model is **Qwen/Qwen2.5-0.5B-Instruct**, a separate causal language model from the MiniLM embedding model. It runs on CPU using Transformers, with a pinned revision, local-only weight loading, and no remote code or inference fallback. Download its approximately 1 GB of weights once:

```powershell
.\.runtime\python.exe app.py --download-generator
```

Set `GENERATION_MODEL`, `GENERATION_REVISION`, and `GENERATION_MAX_NEW_TOKENS` in `.env` to configure it, then restart. Local generation needs several GB of free RAM and can be slower than a hosted model. This small default model prioritizes local CPU use; interpretive quality and citation reliability may be limited. A larger compatible instruction model can be selected if your hardware supports it. The prompt retains current evidence and trims older conversation first when the token budget is exceeded. Pinecone indexing does not need to be repeated when changing only the answer model.

No OpenAI key is required or used. [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

## Pipeline and results

1. **Fetch:** only the two URLs above are configured. The app validates UTF-8 text, Gutenberg START/END markers, and ebook IDs. Existing valid downloads are reused. To download new copies, run `.\.runtime\python.exe app.py --prepare --refresh` while the server is stopped. Raw files retain the Gutenberg license.
2. **Prepare:** anthology bodies are split into 2,100-character windows with 300-character overlap. Metadata includes the volume title, author, ebook ID, source URL, body character offsets, capture time, and normalized document checksum. Body offsets refer to the cleaned text, not printed pages. Editorial material within the anthology body remains available and is distinguished in the answer prompt.
3. **Embed:** the Hugging Face model runs locally on CPU through Sentence Transformers. Each input is split into windows within the model context limit; normalized window vectors are averaged and normalized again. This embeds the full passage without silent truncation. Events expose the model, pinned revision, pooling settings, batch sizes, and 384-dimensional output. There are no OpenAI embedding calls.
4. **Store:** Pinecone upsert receipts are checked. Each run uses a fresh `dgrag-poe-…` namespace; it becomes active only when all batches succeed. Old namespaces remain for audit and manual cleanup. Failed writes preserve the previous active namespace. Pinecone can take a moment to make writes queryable.
5. **Retrieve:** questions use the same local model, revision, normalization, and pooling settings as ingestion and retrieve eight chunks from the active namespace. Old OpenAI indexes and changed model settings are rejected; reindex after switching.
6. **Answer:** the local Hugging Face Qwen model generates an answer constrained to the retrieved text, with numbered citations. The dashboard shows similarity scores and exact passages. It does not guarantee exhaustive analysis across every work. These sources cover only Volumes 1 and 2.

The dashboard includes a searchable volume collection, every document chunk, event details, question history, and **Export results** for full JSON. Provider secrets and full embedding arrays are excluded. The corpus uses `data/poe-state.json`, so previous application data cannot become the active Poe collection.

Optional JSON imports accept one or both configured sources, each with `url` and `text` containing the **full raw Gutenberg file including its markers and ebook header**. Other URLs and duplicates are rejected. Example shape:

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

The unit suite uses mocked model execution and provider responses. Separate embedding and generation checks run the actual cached models on Poe passages with network connections blocked. Receipts are saved to `data/local-embedding-check.json` and `data/local-generation-check.json`. Only Pinecone access requires credentials. Local generation requires downloaded model weights. The application is a single-user local tool. WORK_SUMMARY.md and the companion Page retain a one-page work record.
