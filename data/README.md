# Corpus provenance and sample data

`frames_sample.json` contains two complete query records from the original `OMNIA/dataset/frames/frames_chunks_512.json`. It retains every chunk, the reference answers and the source URL list for each selected query. `manifest.json` records the selection rule, exact sizes, chunk counts and SHA-256 checksums. It is a small service smoke-test sample, not a representative benchmark split or a basis for reporting accuracy or speedup.

## How the experimental corpora were collected

The main `OMNIA/dataset/*/download*.py` scripts obtain questions, answers and page URLs from benchmark records. They fetch those pages through **Jina Reader** (`https://r.jina.ai/<source-url>`), strip Markdown using a custom **Mistune** renderer, concatenate the retrieved text, and save per-question `corpus`, `url_list` and `answer_list` fields. Questions for which all page retrievals fail are skipped.

The `chunking_*.py` scripts tokenize the collected text with **Qwen3-Reranker-4B's tokenizer**, split it into non-overlapping token windows, decode each window, and discard chunks with at most 50 characters after trimming. The project contains 256-, 512- and 1024-token artifacts where available. Window size is measured with the tokenizer, not by words or characters. The included sample is copied from the existing 512-token artifact; it was not fetched or re-tokenized during this cleanup.

| Dataset | Question and URL source used by the original script |
| --- | --- |
| FRAMES | `google/frames-benchmark`, test split; `Prompt`, `Answer` and up to ten `wikipedia_link_*` fields |
| ASQA | `din0s/asqa`, dev split; individual `qa_pairs` questions and short answers, with `wikipages` URLs |
| FreshQA | Local `FreshQA_v112425 - freshqa.csv`; questions, answers and newline-separated `source` URLs |
| HotpotQA | `hotpotqa/hotpot_qa`, distractor validation; sampled questions and Wikipedia URLs derived from supporting-fact titles |
| MS MARCO | `microsoft/ms_marco`, v1.1 test; sampled queries and `passages.url`, excluding empty answer lists |
| SQuAD | `rajpurkar/squad`, validation; sampled questions and Wikipedia URLs derived from article titles |
| TriviaQA | `mandarjoshi/trivia_qa`, rc validation; sampled questions and `search_results.url` |

This provenance matters: these main experiments replay **benchmark-associated pages**, including supporting/reference pages for some datasets. They do not constitute a fresh unrestricted web search for every request. Download success, duplicate question keys and sampling affect the final query counts. The sample preserves the original source URLs for attribution; it does not establish a new license for the underlying pages.

Separate exploratory scripts, particularly `OMNIA/web-to-local-new.py`, use a different route: a local **Qwen3-32B** rewrites a question into several English search queries, **DDGS** supplies result URLs, **Jina Reader** extracts page text, and **LlamaIndex SentenceSplitter** produces chunks (512 tokens with 10-token overlap in the active implementation). Do not attribute all `OMNIA/dataset` artifacts to this search-based route. The small `OMNIA/en_jina_corpus.json` file lacks per-record URL provenance, so the packaged sample instead comes from FRAMES.

The descriptions above were checked against the collection and chunking code in the original workspace. Historical remote page contents and package versions are not recoverable from those scripts alone; a fresh crawl may differ from the saved artifacts.

## Why the full corpora are not copied here

The seven existing 512-token JSON files total **289,638,696 bytes (276.22 MiB)**. Keeping the raw text and alternative chunk sizes would increase this further. The full artifacts remain in the parent workspace rather than being duplicated in this compact repository.

| Original file under `OMNIA/dataset/` | Queries | Size (MiB) |
| --- | ---: | ---: |
| `asqa/asqa_chunks_512.json` | 1097 | 78.01 |
| `frames/frames_chunks_512.json` | 243 | 24.23 |
| `freshqa/freshqa_chunks_512.json` | 419 | 33.28 |
| `hotpotqa/hotpotqa_chunks_512.json` | 328 | 16.01 |
| `msmarco/msmarco_chunks_512.json` | 959 | 85.77 |
| `squad/squad_chunks_512.json` | 188 | 21.54 |
| `triviaqa/triviaqa_chunks_512.json` | 367 | 17.38 |

## Using a saved corpus

Set `dataset_path` in the root `config.json` to `data/frames_sample.json`. For the full FRAMES artifact in the original workspace, use `../OMNIA/dataset/frames/frames_chunks_512.json`. Paths are resolved relative to the configuration file, not the shell's working directory.

The service expects this shape:

```json
{
  "Question text": {
    "chunks": ["First passage", "Second passage"],
    "answer_list": ["Reference answer"],
    "url_list": ["https://example.org/source"]
  }
}
```

Only `chunks` is passed to the model; reference answers and URL metadata are preserved for provenance and evaluation. A request without `chunks` uses an exact question lookup. An explicit `chunks` list overrides the saved corpus. Omnia performs no network search or crawling in the serving path.

After enabling the sample in `config.json` and starting the service, run from the repository root:

```bash
python - <<'PY'
import json
import urllib.request
from pathlib import Path

corpus = json.loads(Path("data/frames_sample.json").read_text())
payload = {"request": next(iter(corpus)), "top_k": 5, "mode": "omnia"}
request = urllib.request.Request(
    "http://127.0.0.1:9876/generate",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=600) as response:
    print(json.dumps(json.load(response), indent=2))
PY
```
