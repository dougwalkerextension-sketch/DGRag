# DGRag application work summary

Updated October 21, 2026. DGRag is a local research application using only Project Gutenberg [Poe Volume 1, ebook 2147](https://www.gutenberg.org/cache/epub/2147/pg2147.txt) and [Volume 2, ebook 2148](https://www.gutenberg.org/cache/epub/2148/pg2148.txt).

## Completed

Built a Python server and responsive dashboard with searchable source text, chunk inspection, pipeline events, provider receipts, JSON export, questions, and source passages. Replaced the original disc golf source with validated Gutenberg downloads, caching, ebook and body-marker checks, wrapper removal, checksums, and volume/character-offset metadata. Both files contain 1,173,212 body characters and produce **653 overlapping chunks**.

Added **Ask Poe**, a labeled literary simulation with follow-up conversations, topic starters for characters, feelings, themes, ravens and clocks, and expandable supporting passages. Browser history retains up to 20 exchanges; recent exchanges supply context. New conversation resets the browser dialogue while the application retains its answer audit log. Keyboard tabs and a direct `/#ask-poe` link are supported.

Embedding inference runs locally on CPU using pinned **sentence-transformers/all-MiniLM-L6-v2**, producing 384-dimensional normalized vectors. Long passages are split into token windows and pooled. Indexing and retrieval enforce matching model settings. Pinecone uses fresh namespaces, activating only fully acknowledged ingestions.

Replaced remote OpenAI answer generation with pinned **Qwen/Qwen2.5-0.5B-Instruct**, running locally through Transformers. Downloaded its weights into the workspace cache. Both generation and embeddings use local-only loading and no remote code; no OpenAI key or API call is used. Generation preserves current evidence and trims older history when needed. The small CPU model can be slower and less reliable than larger models; supporting passages remain inspectable.

Resolved Pinecone network sandbox and incompatible hosted-embedding index errors. Created **dgrag-poe-local**, a 384-dimensional cosine vector index, preserved the original index, updated `.env`, and verified all **653 vectors** and live retrieval. Added a reusable configuration script and clearer recovery messages.

## Verification and operation

All **23 automated tests pass**. Separate real-model checks verified local embeddings and answer generation with network connections blocked; receipts are saved under `data/`. Live Ask Poe HTTP requests completed using Pinecone passages and local Qwen generation. Pinecone storage and retrieval are verified; browser rendering remains unverified.

Run `.\.runtime\python.exe app.py` from a network-enabled terminal and open http://127.0.0.1:8000/#ask-poe. Pinecone credentials and index host are configured. Both model caches are installed. Fresh checkouts use `--download-model` and `--download-generator`; README.md explains configuration. Changing only the answer model requires a restart, not Pinecone reindexing. Keep this summary and its companion Page current.

Prepared a GitHub source package with 16 source/configuration/test files. Expanded ignore rules for environment variants, local tooling, and build output. The configured-secret scan passed; credentials, model weights, runtime binaries, and local data are excluded. `dist/DGRag-source.zip` is ready. GitHub publication is pending account connection; no remote repository has been created.
