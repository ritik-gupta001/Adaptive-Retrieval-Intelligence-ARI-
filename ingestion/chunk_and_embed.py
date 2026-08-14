import sys
import shutil
from pathlib import Path
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    import pypdf as pdf_lib
except ImportError:
    import PyPDF2 as pdf_lib

CHROMA_DIR = "./data/chroma"
COLLECTION_NAME = "ari_default"


def read_file_content(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        text = ""
        try:
            reader = pdf_lib.PdfReader(str(path), strict=False)
            num_pages = len(reader.pages)
            print(f"  Reading '{path.name}' ({num_pages} pages)...")
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
            print(f"    Loaded '{path.name}' ({len(content)} characters).")
        else:
            print(f"    Skipping '{path.name}' (0 readable text characters).")
    return docs


def chunk_and_embed(folder: str, chunk_size: int = 800, chunk_overlap: int = 120):
    raw_docs = load_documents(folder)
    if not raw_docs:
        print(f"No documents (.pdf, .txt, .md) found in '{folder}' — nothing to index.")
        return

    print("Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    documents = []
    ids = []
    metadatas = []
    
    chunk_idx = 0
    for d in raw_docs:
        chunks = splitter.split_text(d["content"])
        print(f"  Split '{d['source']}' into {len(chunks)} chunks.")
        for chunk in chunks:
            documents.append(chunk)
            ids.append(f"doc_{chunk_idx}")
            metadatas.append({"source": d["source"], "path": d["path"]})
            chunk_idx += 1

    print(f"Total chunks generated: {len(documents)}. Connecting to ChromaDB at '{CHROMA_DIR}'...")
    
    # Remove stale sqlite database directory if corrupted
    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists():
        shutil.rmtree(chroma_path, ignore_errors=True)
        print(f"Purged old directory '{CHROMA_DIR}' for clean database initialization.")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    
    # Add to Chroma collection in batches of 1000
    batch_size = 1000
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_meta)
        print(f"  Indexed batch {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1} ({len(batch_docs)} chunks)...")

    print(f"Successfully ingested {len(documents)} total chunks from {len(raw_docs)} files in '{folder}' into Chroma collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "./data/raw"
    chunk_and_embed(folder)
