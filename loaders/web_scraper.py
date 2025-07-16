from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
from langsmith.run_helpers import traceable  # ✅ LangSmith tracing

@traceable(name="DuckDuckGo Search")
def search_duckduckgo(query, max_results=3):
    """
    Perform DuckDuckGo search and return top result URLs.
    Traced in LangSmith as 'DuckDuckGo Search'.
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(r["href"])
    return results

@traceable(name="Scrape Web URLs")
def scrape_text_from_urls(urls):
    """
    Scrapes visible paragraph text from each URL.
    Traced in LangSmith as 'Scrape Web URLs'.
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
        except Exception:
            continue
    return "\n\n".join(texts) if texts else "No relevant content found."

@traceable(name="Get Web Search Context")
def get_duckduckgo_search_context(query, max_results=3):
    """
    Full pipeline: DuckDuckGo search + scrape content.
    Traced in LangSmith as 'Get Web Search Context'.
    """
    urls = search_duckduckgo(query, max_results=max_results)
    return scrape_text_from_urls(urls)
