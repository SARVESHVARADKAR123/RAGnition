from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests

def search_duckduckgo(query, max_results=3):
    """
    Perform DuckDuckGo search and return top result URLs.
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(r["href"])
    return results

def scrape_text_from_urls(urls):
    """
    Scrapes visible paragraph text from each URL.
    """
    texts = []
    for url in urls:
        try:
            res = requests.get(url, timeout=5)
            soup = BeautifulSoup(res.text, "html.parser")
            paragraphs = soup.find_all("p")
            combined = " ".join([p.get_text(strip=True) for p in paragraphs])
            if combined:
                texts.append(combined)
        except:
            continue
    return "\n\n".join(texts) if texts else "No relevant content found."

def get_duckduckgo_search_context(query, max_results=3):
    """
    Main function to get context from top DuckDuckGo results.
    """
    urls = search_duckduckgo(query, max_results=max_results)
    return scrape_text_from_urls(urls)
