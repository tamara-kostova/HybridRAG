from typing import List
from sentence_transformers import SentenceTransformer

class DocumentProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def chunk_text(self, text, max_chunk_size=1000) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            if current_size + len(word) > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def embed_text(self, text):
        return self.model.encode(text)

    def process_paper(self, paper):
        full_text = f"{paper['title']} {paper['abstract']}"
        chunks = self.chunk_text(full_text)
        embeddings = [self.embed_text(chunk) for chunk in chunks]
        
        return {
            "pmid": paper["pmid"],
            "chunks": chunks,
            "embeddings": embeddings
        }
