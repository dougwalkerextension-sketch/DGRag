# DGRag application work summary

Updated October 21, 2026. Source repository: https://github.com/dougwalkerextension-sketch/DGRag.

## Sources and simulator

DGRag is a local research dashboard and labeled Poe literary simulation using three configured sources: Project Gutenberg [Volume 1, ebook 2147](https://www.gutenberg.org/cache/epub/2147/pg2147.txt), [Volume 2, ebook 2148](https://www.gutenberg.org/cache/epub/2148/pg2148.txt), and the Poe Society's [The Philosophy of Composition (1846), Text-02](https://www.eapoe.org/works/essays/philcomp.htm).

The essay frames Ask Poe's interpretations through intended effect, planning the ending, unity of impression, and deliberate choices of tone, beauty, melancholy, sound, refrain, contrast, and suggestiveness. Replies distinguish Poe's published account from invented private thoughts and distinguish interpretations of other works from explicit statements. Each conversation turn retrieves up to two essay passages and six anthology passages with separate Pinecone filters. Essay passages appear first in the numbered evidence and have inspectable links. Research mode retrieves eight passages across all sources. Missing essay evidence blocks simulator generation.

## Implementation

The Python server and responsive dashboard provide searchable source bodies, chunk inspection, pipeline events, receipts, JSON export, source citations, and follow-up conversations. Browser history retains up to 20 exchanges; the most recent six supply conversational context rather than evidence. Keyboard tabs and the direct `/#ask-poe` link are supported.

Gutenberg downloads validate ebook IDs and body markers. The essay loader validates its HTML text container, identifying text, length, and ending; it excludes navigation, scripts, styles, page labels, and external editorial notes. Raw downloads remain under ignored `data/`. The live essay produced 27,096 cleaned characters and 16 chunks. A cache-only integration check loaded all three real sources and produced 669 chunks, including the existing 653 anthology chunks. Chunks use 2,100 characters with 300 overlap and retain source type, title, ID, author, URL, body offsets, capture time, and checksum.

Pinned MiniLM embeddings run locally on CPU with 384-dimensional normalized vectors and pooled token windows. Pinned Qwen2.5-0.5B-Instruct generates answers locally with no remote code or hosted inference. Pinecone storage uses fresh namespaces and activates only fully acknowledged ingestions. Model settings must match for retrieval. Current essay framing and evidence remain subject to the small local model's quality limits; citations should be inspected.

The three-source corpus version and indexed source URLs are recorded only on successful indexing. Older two-volume indexes require loading and reindexing. Preparing texts alone preserves the active index. Partial imports support research, while Ask Poe requires an indexed essay. Previous namespaces remain available; none are deleted.

## Verification and operation

All 26 automated tests pass, covering source validation, three-source caching, extraction exclusions, chunk provenance, failed writes, old-index migration, missing essay evidence, reserved essay retrieval, framing instructions, conversation history, and local inference settings. The actual essay HTML was downloaded and parsed successfully. Historical real-model checks verified offline inference and live Pinecone retrieval for the two-volume version. This update does not claim a new live three-source Pinecone ingestion or browser rendering verification.

Restart the server, open http://127.0.0.1:8000, select **Load Poe texts**, then **Index in Pinecone** to activate all three sources. Run from a network-enabled terminal using `.\.runtime\python.exe app.py`. Fresh checkouts need Python dependencies, both model downloads, and Pinecone configuration as described in README.md. Changing only the answer model requires a restart rather than reindexing.

The source package includes 17 files and passes a scan against configured secrets. `.env`, local data, downloaded models, and runtime binaries are excluded from publication. The repository contains source code; it does not host the running application.
