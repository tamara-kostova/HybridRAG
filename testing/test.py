import requests
import time

BASE_URL = "http://127.0.0.1:8000"


def test_ingest():
    ingest_url = f"{BASE_URL}/ingest"
    ingest_query = {"text": "neurology"}

    print("Testing /ingest")
    response = requests.post(ingest_url, json=ingest_query)

    if response.status_code == 200:
        print("Ingestion successful:")
        print(response.json())
    else:
        print(f"Ingestion failed with status code {response.status_code}:")
        print(response.text)

    print("\nWaiting for 5 seconds for ingestion to complete...\n")
    time.sleep(5)


def test_query():
    query_url = f"{BASE_URL}/query"
    sample_query = {"text": "What are the latest findings in Alzheimer's research?"}

    print("Testing /query ")
    response = requests.post(query_url, json=sample_query)

    if response.status_code == 200:
        print("Query successful:")
        results = response.json()
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"ID: {result['id']}")
            print(f"Score: {result['score']}")
            print(f"Text: {result['text'][:100]}...")
    else:
        print(f"Query failed with status code {response.status_code}:")
        print(response.text)


if __name__ == "__main__":
    # test_ingest()
    test_query()
