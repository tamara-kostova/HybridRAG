from typing import Dict, List
import requests
from bs4 import BeautifulSoup
import time

class PubMedScraper:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.search_url = f"{self.base_url}esearch.fcgi"
        self.fetch_url = f"{self.base_url}efetch.fcgi"

    def search_papers(self, query, max_results=100) -> List[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        response = requests.get(self.search_url, params=params)
        print(response)
        data = response.json()
        return data["esearchresult"]["idlist"]

    def fetch_paper(self, pmid) -> Dict[str, str]:
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml"
        }
        response = requests.get(self.fetch_url, params=params)
        soup = BeautifulSoup(response.content, "lxml-xml")
        
        title = soup.find("ArticleTitle").text if soup.find("ArticleTitle") else ""
        abstract = soup.find("AbstractText").text if soup.find("AbstractText") else ""
        
        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract
        }

    def scrape_papers(self, query, max_results=100) -> List[Dict[str, str]]:
        paper_ids = self.search_papers(query, max_results)
        papers = []
        
        for pmid in paper_ids:
            paper = self.fetch_paper(pmid)
            papers.append(paper)
            time.sleep(1)
        
        return papers