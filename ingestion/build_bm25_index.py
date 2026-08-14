import json
import sys
from pathlib import Path

try:
    import pypdf as pdf_lib
except ImportError:
    import PyPDF2 as pdf_lib

from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(".")

OUTPUT_PATH = Path("./data/bm25_corpus.json")


def read_file_content(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        text = ""
        try:
            reader = pdf_lib.PdfReader(str(path), strict=False)
            num_pages = len(reader.pages)
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception:
                    pass
        except Exception as exc:
            print(f"Warning: Failed to read PDF {path}: {exc}")
        return text
    else:
        return path.read_text(errors="ignore")


CORE_BOOKS = {f"book-{i}.pdf" for i in range(1, 6)}


def load_documents(folder: str):
    docs = []
    folder_path = Path(folder)
    if not folder_path.exists():
        return docs

    paths = [
        p for p in folder_path.rglob("*")
        if p.is_file() and p.suffix.lower() in (".txt", ".md", ".pdf")
        and (p.suffix.lower() != ".pdf" or p.name.lower() in CORE_BOOKS)
    ]
    paths.sort(key=lambda p: p.name)
    print(f"Found {len(paths)} core files to process in '{folder}'.")

    for path in paths:
        content = read_file_content(path)
        if content and content.strip():
            docs.append({"content": content, "source": path.name, "path": str(path)})
    return docs


def build_bm25_index(folder: str, chunk_size: int = 800, chunk_overlap: int = 120):
    raw_docs = load_documents(folder)
    if not raw_docs:
        print(f"No documents (.pdf, .txt, .md) found in '{folder}' — nothing to index.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    corpus = []
    for d in raw_docs:
        for chunk in splitter.split_text(d["content"]):
            corpus.append({"content": chunk, "source": d["source"], "metadata": {"source": d["source"]}})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

    print(f"Wrote {len(corpus)} chunks from {len(raw_docs)} files to {OUTPUT_PATH}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "./data/raw"
    build_bm25_index(folder)
