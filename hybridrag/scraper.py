from typing import Dict, List
import requests
from bs4 import BeautifulSoup
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PubMedScraper:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.search_url = f"{self.base_url}esearch.fcgi"
        self.fetch_url = f"{self.base_url}efetch.fcgi"

    def search_papers(self, query, max_results=100) -> List[str]:
        try:
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
            }
            response = requests.get(self.search_url, params=params)
            print(response)
            data = response.json()
            print(data)
            return data["esearchresult"]["idlist"]
        except requests.RequestException as e:
            logger.error(f"Error searching papers: {e}")
            return []
        except KeyError as e:
            logger.error(f"Unexpected response format: {e}")
            return []

    def fetch_paper(self, pmid) -> Dict[str, str]:
        try:
            params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
            response = requests.get(self.fetch_url, params=params)
            soup = BeautifulSoup(response.content, "lxml-xml")

            title = soup.find("ArticleTitle").text if soup.find("ArticleTitle") else ""
            abstract = (
                soup.find("AbstractText").text if soup.find("AbstractText") else ""
            )

            return {"pmid": pmid, "title": title, "abstract": abstract}
        except requests.RequestException as e:
            logger.error(f"Error fetching paper {pmid}: {e}")
            return {"pmid": pmid, "title": "", "abstract": ""}
        except AttributeError as e:
            logger.error(f"Error parsing paper {pmid}: {e}")
            return {"pmid": pmid, "title": "", "abstract": ""}

    def scrape_papers(self, query, max_results=100) -> List[Dict[str, str]]:
        papers = []
        try:
            paper_ids = self.search_papers(query, max_results)
            for pmid in paper_ids:
                paper = self.fetch_paper(pmid)
                papers.append(paper)
                time.sleep(1)
        except Exception as e:
            logger.error(f"Error scraping papers: {e}")
        return papers
