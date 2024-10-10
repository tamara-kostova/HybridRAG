import requests
from bs4 import BeautifulSoup
import time


class PubMedScraper:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.search_url = f"{self.base_url}esearch.fcgi"
        self.fetch_url = f"{self.base_url}efetch.fcgi"

    def search_papers(self, query, max_results=100):
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
        }
        response = requests.get(self.search_url, params=params)
        print(f"REsponse: {response}")
        data = response.json()
        return data["esearchresult"]["idlist"]

    def fetch_paper(self, pmid):
        params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
        response = requests.get(self.fetch_url, params=params)
        soup = BeautifulSoup(response.content, "lxml-xml")

        title = soup.find("ArticleTitle").text if soup.find("ArticleTitle") else ""
        abstract = soup.find("AbstractText").text if soup.find("AbstractText") else ""

        return {"pmid": pmid, "title": title, "abstract": abstract}

    def scrape_papers(self, query, max_results=100):
        paper_ids = self.search_papers(query, max_results)
        papers = []

        for pmid in paper_ids:
            paper = self.fetch_paper(pmid)
            print(paper)
            papers.append(paper)
            time.sleep(1)

        return papers

if __name__ == "__main__":
    scraper = PubMedScraper()
    papers = scraper.scrape_papers("neurology", max_results=10)
