import re
import requests
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os
import time
from src.legal_assistant.centralized_logger import CentralizedLogger
import sys

from src.legal_assistant.similarity.rank_similarity import get_cosine_similarity_score
from src.legal_assistant.utils.clean_html import clean_legal_html

logger = CentralizedLogger().get_logger()

def extract_case_titles_from_google(results: any) -> set:
    """Extract case titles from Google search results"""

    case_titles = set()

    logger.info("\nExtracted Case Titles:")
    for result in results:
        case_titles.add(result['title'])

    return case_titles

def get_case_titles_from_google(query, original_query, max_results) -> set:
    """Enhanced Google Custom Search API implementation with pagination and error handling"""

    logger.info(f"Searching for: {query}")

    google_api_key = os.getenv("GOOGLE_SEARCH_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    all_scored_docs = []  # To store scored docs from all pages
    max_results = min(max_results, 100)  # Limit to 100 results due to pagination

    for start_index in range(1, max_results + 1, 10):  # Loop through pages of 10 results
        params = {
            'key': google_api_key,
            'cx': search_engine_id,
            'q': query,
            'num': min(10, max_results - start_index + 1),  # Adjust 'num' for the last page
            'safe': 'active',
            'fields': 'items(title, link, snippet)',
            'start': start_index,  # Add the 'start' parameter for pagination
        }

        try:
            response = requests.get(
                'https://www.googleapis.com/customsearch/v1',
                params=params,
                timeout=15
            )
            response.raise_for_status()

            results = response.json()

            if 'error' in results:
                error = results['error']
                code = error.get('code', 400)
                message = error.get('message', 'Unknown API error')
                raise requests.exceptions.HTTPError(f"API Error {code}: {message}")

            if not results.get('items'):
                logger.info("No items found in the response for this page.")
                continue  # Go to the next page

            scored_docs = []

            for item in results['items']:
                docid = extract_docid(item['link'])
                snippet = item['snippet']
                if snippet:
                    snippet = clean_legal_html(snippet)
                cosine_score: float = get_cosine_similarity_score(query, snippet)
                logger.info(f"Cosine score: {cosine_score} for case title: {item['title']}")
                if docid:
                    scored_docs.append((docid, cosine_score))

            all_scored_docs.extend(scored_docs) # add results from the current page to the all_scored_docs list

            time.sleep(0.1)  # Respect API rate limits

        except requests.exceptions.RequestException as e:
            logger.error(f"Search failed: {str(e)}")
            return set()
        except ValueError as ve:
            logger.error(f"JSON parsing error: {str(ve)}")
            return set()

    # Sort all documents by cosine similarity score in descending order
    all_scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Extract the top 5 docids
    top_docids = [doc[0] for doc in all_scored_docs[:5]]

    logger.info(f"The top 5 docids with highest cosine similarity scores are: {top_docids}")

    return top_docids

def extract_docid(url):
    """
    Extracts the docid from the given URL.

    Args:
        url (str): The input URL string.

    Returns:
        str: The extracted docid, or None if no match is found.
    """
    match = re.search(r'/doc/(\d+)', url)
    if match:
        return match.group(1)
    return None

if __name__ == "__main__":
    load_dotenv()

    google_api_key = os.getenv("GOOGLE_SEARCH_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    try:
        results = get_case_titles_from_google(
            query="can anticipatory bail be granted in a murder case",
            original_query="can anticipatory bail be granted in a murder case",
            max_results=100  # Request 100 results
        )

        logger.info(f"Top DocIDs: {results}")

    except ValueError as ve:
        logger.info(f"Configuration Error: {str(ve)}")
    except Exception as e:
        logger.info(f"Unexpected Error: {str(e)}")