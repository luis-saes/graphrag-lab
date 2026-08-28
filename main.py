import os
import pickle
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
import numpy as np
import time
from google.genai import errors

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-3.1-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
MAX_OUTPUT_TOKENS = 500
EMBED_CHUNK_SIZE = 50
TRANSCRIPTIONS_PATH = "./transcriptions"
CHUNKS_PATH = "./chunks/chunks.pkl"

client = genai.Client(api_key=GEMINI_API_KEY)
embed_model = SentenceTransformer(EMBED_MODEL)

questions = [
    "Who created FastAPI and is now building a cloud for it?",
    "Datastar's whole pitch is that it's tiny. How tiny?",
    "David Flood does digital humanities work at which university?",
    "How many people sit on the Python Typing Council?",
    "Which open source project is the million-line monorepo they tour?",
    "Monty is a Python interpreter written in what language?",
    "Before Zensical, Martin Donath was best known for which project?",
    "Deep Agents comes from which company?",
    "Compiled wheels are stuck targeting CPU features from roughly what year?",
    "Tanya Janca is there to walk through which famous list?",
    "Alex Kretzschmar is head of DevRel at which company?",
    "Ray came out of which lab, at which university?",
    "Chris May's go-to analogy for event sourcing is which version control system?",
    "Rich Iannone and Michael Chow both work for which company?",
    "Paolo's opening horror story is a pull request with how many lines added?",
    "At PyCon, Startup Row lives in which part of the venue?",
    "Which company acquired Astral?",
    "Who replaced Brian Okken as co-host of Python Bytes?",
    "Which city is Sumit Gundawar based in?",
    "marimo pair puts a coding agent inside what, specifically?",
]


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


def embed_texts(texts: list[str]) -> list[list[float]]:
    return embed_model.encode(texts, normalize_embeddings=True).tolist()


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
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=max_tokens),
            )
            return resp.text
        except errors.ServerError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


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
                    extract_texts_from_chunks(chunks[i : i + EMBED_CHUNK_SIZE])
                )
                update_chunks_with_embeddings(chunks, temp_embeddings, i)
                print(
                    "Chunk",
                    i / EMBED_CHUNK_SIZE,
                    "out of",
                    len(chunks) / EMBED_CHUNK_SIZE,
                )

            chunks_path.parent.mkdir(parents=True, exist_ok=True)

            with chunks_path.open("wb") as f:
                pickle.dump(chunks, f)

        embedded_questions = embed_texts(questions)

        scores_per_question = []
        for i, question in enumerate(embedded_questions):
            scores_per_question.append([])
            for chunk in chunks:
                scores_per_question[i].append(np.dot(question, chunk.embedding))

        top_fives = []
        for result in scores_per_question:
            top_fives.append(np.argsort(np.array(result))[::-1][:5])

        for i, top_five in enumerate(top_fives):
            print("Question:", questions[i])
            for answer_index in top_five:
                print("Score:", scores_per_question[i][answer_index])
                print("Answer:", chunks[answer_index].text)

        separator = "\n\n---\n\n"
        for i, question in enumerate(questions):
            five_joined = ""
            for top_five in top_fives[i]:
                five_joined += chunks[top_five].text + separator
            text = complete(
                "Based on this question '"
                + question
                + "'"
                + "using only the context that I'll send you, find answer to the question. If the answer is not there, tell so. Context:'"
                + five_joined
                + "'."
            )
            print("response:", text)

    finally:
        client.close()
