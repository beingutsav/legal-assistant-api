import re
import requests
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os
import time
import logging

log_dir = os.path.expanduser("~/log")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "search_calls.log")


logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def extract_case_titles_from_google(results: any) -> set:
        """Extract case titles from Google search results"""

        case_titles = set()

        logger.info("\nExtracted Case Titles:")
        for result in results:
            case_titles.add(result['title'])

        
        return case_titles


def get_case_titles_from_google(query, max_results) -> set:
    """Enhanced Google Custom Search API implementation with better error handling"""
    
    logger.info(f"Searching for: {query}")

    google_api_key = os.getenv("GOOGLE_SEARCH_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    # API parameters as per official documentation
    params = {
        'key': google_api_key,
        'cx': search_engine_id,
        'q': {query},
        'num': min(max_results, 10),  # API maximum is 10 results per request
        'safe': 'active',  # Enable safe search
        'fields': 'items(title, link)',  # Request only necessary fields
    }

    try:
        response = requests.get(
            'https://www.googleapis.com/customsearch/v1',
            params=params,
            timeout=15
        )
        response.raise_for_status()
        
        results = response.json()

        logger.info(f"the response is .. {results}")
        
        # Check for API errors in JSON response
        if 'error' in results:
            error = results['error']
            code = error.get('code', 400)
            message = error.get('message', 'Unknown API error')
            raise requests.exceptions.HTTPError(f"API Error {code}: {message}")
        

        caseDocs = set();

        for item in results['items']:
            docid = extract_docid(item['link'])
            if docid:
                caseDocs.add(docid)

        #return process_results(results.get('items', []), max_results)
        logger.info(f"the docids extracted are {caseDocs}")
        return caseDocs;
        
    except requests.exceptions.RequestException as e:
        logger.info(f"Search failed: {str(e)}")
        return []
    except ValueError as ve:
        logger.info(f"JSON parsing error: {str(ve)}")
        return []
    

def extract_docid(url):
    """
    Extracts the docid from the given URL.

    Args:
        url (str): The input URL string.

    Returns:
        str: The extracted docid, or None if no match is found.
    """
    # Use a regular expression to find the numeric docid in the URL
    match = re.search(r'/doc/(\d+)', url)
    if match:
        return match.group(1)  # Extract and return the first capturing group (docid)
    return None  # Return None if no docid is found
    
    

if __name__ == "__main__":
    load_dotenv()
    
    google_api_key = os.getenv("GOOGLE_SEARCH_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    # Example search with error handling
    try:
        results = get_case_titles_from_google(
            query="can anticipatory bail be granted in a murder case",
            max_results=5
        )

        case_titles = set()

        logger.info("\nExtracted Case Titles:")
        for result in results:
            case_titles.add(result['title'])

            
    except ValueError as ve:
        logger.info(f"Configuration Error: {str(ve)}")
    except Exception as e:
        logger.info(f"Unexpected Error: {str(e)}")