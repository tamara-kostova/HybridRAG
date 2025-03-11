import glob
import os
import sys
from src.db.db_client import QdrantWrapper

if len(sys.argv) != 3:
    print("Usage: python -m testing.filter_relevant_papers <QDRANT_HOST> <QDRANT_API_KEY>")
    sys.exit(1)

RELEVANT_PDFS_PATH = "selected_pdfs"

qdrant_host = sys.argv[1]
qdrant_api_key = sys.argv[2]

qdrant_wrapper = QdrantWrapper(
    url=qdrant_host, api_key=qdrant_api_key
)

collection_info = qdrant_wrapper.get_collection_info()
print("Collection Info:", collection_info)

# papers = qdrant_wrapper.fetch_all_documents_in_batches()
# all_paper_names = set()
# for paper in papers:
#     all_paper_names.add(paper["paper_name"])
# for paper_name in all_paper_names:
#     with open("papers.txt", "a") as f:
#         f.write(f"{paper_name}\n")

with open("papers.txt", "r") as f:
    papers = f.read().splitlines()

print(f"{len(papers)} unique papers")
all_papers = set(papers)
pdf_files = glob.glob(os.path.join(RELEVANT_PDFS_PATH, "*.pdf"))

with open("selected_pdfs.txt", "w") as f:
    for pdf_file in pdf_files:
        filename = os.path.basename(pdf_file)
        f.write(f"{filename}\n")

selected_papers = set(pdf_files)
papers_to_delete = all_papers - selected_papers
print(f"Will delete {len(papers_to_delete)} papers that are not in the selected list")

deleted_count = 0
for paper_name in papers_to_delete:
    try:
        count = qdrant_wrapper.delete(paper_name=paper_name)
        deleted_count += count
        print(f"Deleted paper: {paper_name} ({count} chunks)")
    except Exception as e:
        print(f"Error deleting paper {paper_name}: {e}")

print(f"Deletion complete. Deleted {deleted_count} chunks from {len(papers_to_delete)} papers.")