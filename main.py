import os
import pickle
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-3.1-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
MAX_OUTPUT_TOKENS = 500
EMBED_CHUNK_SIZE = 50
TRANSCRIPTIONS_PATH = "./transcriptions"
CHUNKS_PATH = "./chunks/chunks.pkl"

client = genai.Client(api_key=GEMINI_API_KEY)


@dataclass
class Chunk:
    text: str
    chunk_index: int
    start: int
    end: int
    source_path: Path
    source_filename: str
    embedding: list[float] | None = None


def load_files_path(path: str) -> list[Path]:
    folder = Path(path)
    files = [path for path in folder.iterdir() if path.is_file()]
    return files


def load_file(file_path: Path) -> str:
    content = file_path.read_text(encoding="utf-8")
    return content


def chunk_text(content: str, path: Path) -> list[Chunk]:
    chunk_size_in_characters = 1000
    chunk_overlap_in_characters = 150
    last_chunk_used_in_characters = 0
    chunks: list[Chunk] = []

    i = 0
    while last_chunk_used_in_characters < len(content):
        start = 0
        end = 0

        if last_chunk_used_in_characters == 0:
            start = 0
            end = chunk_size_in_characters
        else:
            start = last_chunk_used_in_characters - chunk_overlap_in_characters
            end = start + chunk_size_in_characters

        if end > len(content):
            end = len(content)

        last_chunk_used_in_characters = end

        if end - start <= chunk_overlap_in_characters:
            continue

        chunks.append(
            Chunk(
                text=content[start:end],
                chunk_index=i,
                start=start,
                end=end,
                source_path=path,
                source_filename=path.name,
            )
        )
        i += 1

    return chunks


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embed_model.encode(texts).tolist()


def extract_texts_from_chunks(chunks: list[Chunk]) -> list[str]:
    texts: list[str] = []
    for chunk in chunks:
        texts.append(chunk.text)

    return texts


def update_chunks_with_embeddings(
    chunks: list[Chunk], embeddings: list[list[float]], start: int = 0
):
    for i, embedding in enumerate(embeddings):
        chunks[start + i].embedding = embedding


def complete(prompt: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> str:
    resp = client.models.generate_content(
        model=GEMINI_MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=max_tokens),
    )
    return resp.text


if __name__ == "__main__":
    try:
        chunks_path = Path(CHUNKS_PATH)
        chunks: list[Chunk] = []

        if chunks_path.exists():
            with chunks_path.open("rb") as f:
                chunks = pickle.load(f)
            print("loaded", len(chunks), "chunks from cache")

        else:
            files_path = load_files_path(TRANSCRIPTIONS_PATH)

            for file_path in files_path:
                file_content = load_file(file_path)
                cs = chunk_text(file_content, file_path)

                chunks += cs

            for i in range(0, len(chunks), EMBED_CHUNK_SIZE):
                temp_embeddings = embed_texts(
                    extract_texts_from_chunks(chunks[i : i + EMBED_CHUNK_SIZE]),
                    "RETRIEVAL_DOCUMENT",
                )
                update_chunks_with_embeddings(chunks, temp_embeddings, i)
                print("Chunk", i, "out of", len(chunks) / EMBED_CHUNK_SIZE)

            chunks_path.parent.mkdir(parents=True, exist_ok=True)

            with chunks_path.open("wb") as f:
                pickle.dump(chunks, f)
        print(len(chunks), len(chunks[0].embedding))
    finally:
        client.close()
