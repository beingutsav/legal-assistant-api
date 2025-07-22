import re
import requests
from urllib.parse import quote_plus, urlparse
from dotenv import load_dotenv
import os
import time
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
import hashlib

from src.legal_assistant.centralized_logger import CentralizedLogger
from src.legal_assistant.similarity.rank_similarity import get_cosine_similarity_score
from src.legal_assistant.utils.clean_html import clean_legal_html
from collections import defaultdict


logger = CentralizedLogger().get_logger()

@dataclass
class SearchResult:
    """Structured search result with metadata"""
    url: str
    title: str
    snippet: str
    domain: str
    doc_id: Optional[str] = None
    similarity_score: float = 0.0
    content: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

class DomainProcessor:
    """Base class for domain-specific content processing"""
    
    def __init__(self, domain: str):
        self.domain = domain
    
    def can_process(self, url: str) -> bool:
        """Check if this processor can handle the given URL"""
        return self.domain in url
    
    def extract_doc_id(self, url: str) -> Optional[str]:
        """Extract document ID from URL if applicable"""
        return None
    
    def fetch_content(self, url: str, headers: Dict = None) -> Optional[str]:
        """Fetch and clean content from URL"""
        try:
            logger.info(f"Fetching content with generic processor from: {url}")
            response = requests.get(url, headers=headers or {}, timeout=15)
            response.raise_for_status()
            return self.clean_content(response.text)
        except Exception as e:
            logger.error(f"Failed to fetch content from {url}: {str(e)}")
            return None
    
    def clean_content(self, html: str) -> str:
        """Clean HTML content for this domain"""
        return clean_legal_html(html)
    
    def extract_metadata(self, html: str, url: str) -> Dict:
        """Extract domain-specific metadata"""
        return {}

class IndianKanoonProcessor(DomainProcessor):
    """Processor for Indian Kanoon domain"""
    
    def __init__(self):
        super().__init__("indiankanoon.org")
        self.api_key = os.getenv("INDIAN_KANOON_API_KEY")
        self.doc_url = "https://api.indiankanoon.org/doc/"
    
    def extract_doc_id(self, url: str) -> Optional[str]:
        """Extract docid from Indian Kanoon URL"""
        match = re.search(r'/doc/(\d+)', url)
        return match.group(1) if match else None
    
    def fetch_content(self, url: str, headers: Dict = None) -> Optional[str]:
        """Fetch content using Indian Kanoon API"""
        doc_id = self.extract_doc_id(url)
        if not doc_id or not self.api_key:
            return super().fetch_content(url, headers)
        
        try:
            logger.info(f"Fetching content from Indian Kanoon API for doc: {doc_id}")
            api_headers = {"Authorization": f"Token {self.api_key}"}
            response = requests.post(f"{self.doc_url}{doc_id}/", headers=api_headers)
            response.raise_for_status()
            
            data = response.json()
            doc_html = data.get("doc", "")
            return self.clean_content(doc_html)
        except Exception as e:
            logger.error(f"Failed to fetch from Indian Kanoon API for doc {doc_id}: {str(e)}")
            return super().fetch_content(url, headers)
    
    def extract_metadata(self, html: str, url: str) -> Dict:
        """Extract Indian Kanoon specific metadata"""
        doc_id = self.extract_doc_id(url)
        if not doc_id or not self.api_key:
            return {}
        
        try:
            logger.info(f"Extracting metadata from Indian Kanoon API for doc: {doc_id}")
            api_headers = {"Authorization": f"Token {self.api_key}"}
            response = requests.post(f"{self.doc_url}{doc_id}/", headers=api_headers)
            response.raise_for_status()
            
            data = response.json()
            year = data.get("year", "Unknown Year")
            if isinstance(year, float):
                year = int(year)
            
            citations = data.get("citations", [])
            if not isinstance(citations, list):
                citations = [str(citations)]
            
            return {
                "court": data.get("court", "Unknown Court"),
                "year": year,
                "citations": [str(citation) for citation in citations][:3],
                "doc_id": doc_id
            }
        except Exception as e:
            logger.error(f"Failed to extract metadata for doc {doc_id}: {str(e)}")
            return {}

class ManuptraProcessor(DomainProcessor):
    """Processor for Manupatra domain"""
    
    def __init__(self):
        super().__init__("manupatrafast.com")
    
    def clean_content(self, html: str) -> str:
        """Clean Manupatra specific HTML"""
        try:
            logger.info("Cleaning content with ManuptraProcessor")
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove navigation, ads, and other non-content elements
            for element in soup.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style']):
                element.decompose()
            
            # Find main content area (adjust selectors based on Manupatra structure)
            content_selectors = ['.judgment-content', '.case-content', '.main-content', 'article']
            content = None
            
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    break
            
            if not content:
                content = soup.find('body')
            
            return content.get_text(strip=True) if content else soup.get_text(strip=True)
        except Exception as e:
            logger.error(f"Failed to clean Manupatra content: {str(e)}")
            return clean_legal_html(html)

class LawyersClubIndiaProcessor(DomainProcessor):
    """Processor for Lawyers Club India domain"""
    
    def __init__(self):
        super().__init__("lawyersclubindia.com")
    
    def clean_content(self, html: str) -> str:
        """Clean Lawyers Club India specific HTML"""
        try:
            logger.info("Cleaning content with LawyersClubIndiaProcessor")
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove ads, navigation, and other non-content elements
            for element in soup.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style', '.advertisement']):
                element.decompose()
            
            # Find main content
            content_selectors = ['.article-content', '.post-content', '.main-content', 'article']
            content = None
            
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    break
            
            if not content:
                content = soup.find('body')
            
            return content.get_text(strip=True) if content else soup.get_text(strip=True)
        except Exception as e:
            logger.error(f"Failed to clean Lawyers Club India content: {str(e)}")
            return clean_legal_html(html)

class GenericLegalProcessor(DomainProcessor):
    """Generic processor for other legal domains"""
    
    def __init__(self, domain: str):
        super().__init__(domain)
    
    def clean_content(self, html: str) -> str:
        """Generic legal content cleaning"""
        try:
            logger.info(f"Cleaning content with GenericLegalProcessor for domain: {self.domain}")
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove common non-content elements
            for element in soup.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style', '.advertisement', '.ads']):
                element.decompose()
            
            # Try to find main content
            content_selectors = [
                'main', 'article', '.content', '.main-content', 
                '.post-content', '.article-content', '.judgment-content',
                '.case-content', '#content', '#main'
            ]
            
            content = None
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    break
            
            if not content:
                content = soup.find('body')
            
            text = content.get_text(strip=True) if content else soup.get_text(strip=True)
            
            # Clean up extra whitespace
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
            
            return text
        except Exception as e:
            logger.error(f"Failed to clean generic legal content: {str(e)}")
            return clean_legal_html(html)

class EnhancedLegalSearch:
    """Enhanced search system for multiple legal domains"""
    
    def __init__(self):
        self.processors = [
            IndianKanoonProcessor(),
            ManuptraProcessor(),
            LawyersClubIndiaProcessor(),
        ]
        self.generic_processor = GenericLegalProcessor("generic")
        
        # Load Google Search API credentials
        self.google_api_key = os.getenv("GOOGLE_SEARCH_KEY")
        self.search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        
        if not self.google_api_key or not self.search_engine_id:
            raise ValueError("Google Search API credentials not found in environment variables")
    
    def get_processor(self, url: str) -> DomainProcessor:
        """Get appropriate processor for URL"""
        for processor in self.processors:
            if processor.can_process(url):
                return processor
        
        # Use generic processor for unknown domains
        domain = urlparse(url).netloc
        return GenericLegalProcessor(domain)
    
    def search_google_custom(self, search_query: str, semantic_query: str, max_pages: int = 20) -> List[SearchResult]:
        """Search using Google Custom Search API"""
        results = []
        max_pages = min(max_pages, 100)  # API limit
        
        for start_index in range(1, max_pages + 1, 10):
            params = {
                'key': self.google_api_key,
                'cx': self.search_engine_id,
                'q': search_query,
                'num': min(10, max_pages - start_index + 1),
                'safe': 'active',
                'fields': 'items(title, link, snippet)',
                'start': start_index,
            }
            
            try:
                response = requests.get(
                    'https://www.googleapis.com/customsearch/v1',
                    params=params,
                    timeout=15
                )
                response.raise_for_status()
                
                data = response.json()
                
                if 'error' in data:
                    error = data['error']
                    raise requests.exceptions.HTTPError(f"API Error {error.get('code', 400)}: {error.get('message', 'Unknown error')}")
                
                if not data.get('items'):
                    logger.info(f"No items found for page starting at {start_index}")
                    continue
                
                for item in data['items']:
                    url = item['link']
                    domain = urlparse(url).netloc
                    processor = self.get_processor(url)
                    
                    result = SearchResult(
                        url=url,
                        title=item['title'],
                        snippet=item.get('snippet', ''),
                        domain=domain,
                        doc_id=processor.extract_doc_id(url),
                        similarity_score=get_cosine_similarity_score(semantic_query, item.get('snippet', ''))
                    )
                    
                    results.append(result)
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Search failed for page {start_index}: {str(e)}")
                break
        
        return results
    
    def fetch_content_for_results(self, results: List[SearchResult], max_content_fetch: int = 10, max_per_domain: int = 5) -> List[Dict]:
        if not results:
            return []
        
        domain_usage = defaultdict(int)

        # Step 1: Sort results by similarity (high to low)
        sorted_results = sorted(results, key=lambda x: x.similarity_score, reverse=True)

        # Step 2: Try to select top results ensuring domain diversity
        selected_results = []
        already_seen = set()

        for result in sorted_results:
            domain = result.domain
            if domain_usage[domain] < max_per_domain:
                selected_results.append(result)
                domain_usage[domain] += 1
                already_seen.add(result.url)
            if len(selected_results) >= max_content_fetch:
                break

        # Step 3: If still less than max_fetch, fill remaining without domain restriction
        if len(selected_results) < max_content_fetch:
            for result in sorted_results:
                if result.url in already_seen:
                    continue
                selected_results.append(result)
                already_seen.add(result.url)
                if len(selected_results) >= max_content_fetch:
                    break

        # Step 4: Fetch content
        fetched_cases = []
        for result in selected_results:
            processor = self.get_processor(result.url)
            content = processor.fetch_content(result.url)

            if content:
                result.content = content
                result.metadata = processor.extract_metadata(content, result.url)
                logger.info(f"downloaded from url: {result.url}, and the content is .. {content}")
            else:
                logger.warning(f"Failed to fetch content for {result.url}")
        return [result.__dict__ for result in selected_results]
    

    def search_legal_cases(self, search_query: str, semantic_query: str, max_pages: int = 20, max_content_fetch: int = 10, max_per_domain: int = 5) -> List[Dict]:
        """Main search function that returns processed legal cases"""
        logger.info(f"Enhanced search for: {search_query}")
        
        # Step 1: Get search results from Google
        search_results = self.search_google_custom(search_query, semantic_query, max_pages)
        logger.info(f"Found {len(search_results)} search results")
        
        # Step 2: Fetch content for top results
        results_with_content = self.fetch_content_for_results(search_results, max_content_fetch, max_per_domain)
        
        # Step 3: Convert to legacy format for compatibility
        cases = []
        for result in results_with_content:
            case = {
                "id": result.get("doc_id", "") or hashlib.md5(result.get("url", "").encode()).hexdigest()[:8],
                "title": result.get("title", ""),
               # "court": result.get("metadata", {}).get("court", f"Source: {result.get('domain', '')}"),
               # "year": result.get("metadata", {}).get("year", "Unknown Year"),
               # "citations": result.get("metadata", {}).get("citations", []),
                "doc_url": result.get("url", ""),
                "doc_text": result.get("content", ""),
               # "domain": result.get("domain", ""),
                "similarity_score": result.get("similarity_score", 0.0)
            }
            cases.append(case)
        
        logger.info(f"Successfully processed {len(cases)} cases with content")
        logger.info(f"following is the case array .. {cases}")
        return cases

# Convenience function for backward compatibility
def get_enhanced_case_results(optimized_query: str, original_query: str, max_pages: int = 20, max_content_fetch: int = 10, max_per_domain: int = 5) -> List[Dict]:
    """Enhanced version of get_case_titles_from_google with multi-domain support"""
    try:
        search_engine = EnhancedLegalSearch()
        return search_engine.search_legal_cases(optimized_query, original_query, max_pages, max_content_fetch, max_per_domain)
    except Exception as e:
        logger.error(f"Enhanced search failed: {str(e)}")
        return []
