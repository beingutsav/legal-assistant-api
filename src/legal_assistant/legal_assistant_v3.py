import os
import json
import argparse
import requests
from convex import ConvexClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

from src.legal_assistant.centralized_logger import CentralizedLogger
from src.legal_assistant.open_router import OpenRouterLegalAssistant
from src.legal_assistant.similarity.rank_similarity import get_cosine_similarity_score
from .search_google import get_case_titles_from_google
from .sanitize_json import sanitize_to_raw_json
from .prompts.prompts_legal import _doc_summary_prompt, _short_query_prompt, _summary_prompt_template, get_search_prompt_system, get_search_prompt_user, get_search_query_prompt, get_system_prompt2, legal_research_prompt
from .prompts.prompts_legal import get_final_legal_query_resolver_prompt
from .prompts.prompts_legal import _greeting_prompt
from .utils.clean_html import clean_legal_html
from .prompts.prompts_legal import get_system_prompt


# Load environment file from the same package
load_dotenv()


openrouter = OpenRouterLegalAssistant()

logger = CentralizedLogger().get_logger()

# Initialize clients - updated Convex initialization
convex_url = os.getenv("CONVEX_URL")
#convex_key = os.getenv("CONVEX_KEY")

convex = ConvexClient(convex_url)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
generation_config = {
    "temperature": 0.35,    # Increased from 0.2 → 0.35
    "top_p": 0.85,          # Slightly wider sampling
    "top_k": 60,            # Expanded candidate pool
    "max_output_tokens": 3000, # Increased for legal citations
    "frequency_penalty": 1.2, # Add if available
}
model_name = os.getenv("GEMINI_LLM_MODEL")
gemini = genai.GenerativeModel(model_name=model_name, 
                              generation_config=generation_config)

# Indian Kanoon API configuration
INDIAN_KANOON_API_KEY = os.getenv("INDIAN_KANOON_API_KEY")
INDIAN_KANOON_SEARCH_URL = "https://api.indiankanoon.org/search/"
INDIAN_KANOON_DOC_URL = "https://api.indiankanoon.org/doc/"
INDIAN_KANOON_DOC_META_URL = "https://api.indiankanoon.org/docmeta/"

# Maximum number of cases to store in context
MAX_STORED_CASES = 10
# Maximum length of summary for each case
MAX_SUMMARY_LENGTH = 500
# Maximum search results 
MAX_SEARCH_RESULTS_FROM_KANOON = 50

def create_chat_session():
    """Create a new chat session and return its ID"""
    chat_id = f"chat_{os.urandom(4).hex()}"
    convex.mutation("chats:insert", {
        "id": chat_id,
        "summary_context": "",
        "retrieved_cases": [],
    })
    return chat_id

def generate_case_summary(query, full_text, title, max_length=MAX_SUMMARY_LENGTH):
    """Generate a concise summary of a legal case using Gemini"""
   
    generate_case_summary_prompt = _doc_summary_prompt(max_length, query, title, full_text)
    try:
        response = gemini.generate_content(generate_case_summary_prompt)
        summary = response.text.strip()
        # Ensure summary doesn't exceed max length
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        return summary
    except Exception as e:
        logger.info(f"Error generating summary: {e}")
        # Fallback to simple truncation
        return full_text[:max_length-3] + "..."
    


def prepare_search_query(user_query, summary_context):
    """Generate optimized search parameters for Indian Kanoon based on user query and context"""
   # prompt = get_search_query_prompt(user_query, summary_context)
    full_ai_message = generate_search_query_full_prompt(user_query, summary_context)
    
    try:
        response = openRouterResponse(full_ai_message)
        #response = gemini.generate_content(prompt)
        search_query = response
        return search_query
    except Exception as e:
        logger.info(f"Error generating search query: {e}")
        # Fallback to original query if there's an error
        return user_query

def search_indian_kanoon(optimized_query, original_query, pagenum=0, max_results=MAX_SEARCH_RESULTS_FROM_KANOON):
    """Search Indian Kanoon API for relevant cases with optimized storage"""
    logger.info(f"the optimized query is .. {optimized_query}")

    headers = {
        "Authorization": f"Token {INDIAN_KANOON_API_KEY}"
    }

    #remove double quotes from start and end if they exist
    #optimized_query = optimized_query[1:-1]
    
    cases = []
    caseDocs : set = get_case_titles_from_google(optimized_query, original_query, max_results)
    # Convert set to list to allow indexing
    caseDocs_list = list(caseDocs)

    # Process up to max_results cases
    for doc in caseDocs_list[:max_results]:            
        logger.info(f"Fetching document from Indian Kanoon for doc id: {doc}")
        # Get document metadata
        meta_response = requests.post(
            f"{INDIAN_KANOON_DOC_URL}{doc}/", 
            headers=headers
        )
        
        if meta_response.status_code == 200:
            meta = meta_response.json()
            
            # Extract document text
            doc_text_html = meta.get("doc", "")
            doc_text = clean_legal_html(doc_text_html)
            logger.info("cleaned doc is ..")
            logger.info(doc_text)
            
            # Generate a concise summary instead of storing full text
            title = meta.get("title", "Untitled Case")
            
            #dont need summaries, unnecessary calls
            #summary = generate_case_summary(query, doc_text, title)
            
            # Convert year to integer if it's a float
            year = meta.get("year", "Unknown Year")
            if isinstance(year, float):
                year = int(year)
            
            # Ensure citations is an array of strings
            citations = meta.get("citations", [])
            if not isinstance(citations, list):
                citations = [str(citations)]
                
            
            # Store only essential metadata and the summary
            case = {
                "id": str(doc),
                "title": title,
               # "summary": summary,  # This is now a concise summary
                "court": meta.get("court", "Unknown Court"),
                "year": year,
                "citations": [str(citation) for citation in citations][:3],  # Limit citations stored
                "doc_url": f"{INDIAN_KANOON_DOC_URL}{doc}/",
                "doc_text": doc_text  # Store full text for reference
            }
            cases.append(case)
    
    return cases

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return 1 - cosine(a, b)

def get_relevant_cases(query, cases):
    """Find the most relevant cases for a query using embeddings"""
    if not cases:
        return []
        
    query_embed = model.encode(query)
    case_embeds = {c["id"]: model.encode(c["doc_text"]) for c in cases}
    similarities = {cid: cosine_similarity(query_embed, e) for cid, e in case_embeds.items()}
    
    # Sort cases by similarity
    sorted_case_ids = sorted(similarities.items(), key=lambda x: -x[1])

     # Get the top 3 relevant cases with their titles
    top_relevant_cases = [(case_id, score, next(c["title"] for c in cases if c["id"] == case_id)) for case_id, score in sorted_case_ids[:3]]
    
    # logger.info the titles of the top 3 relevant cases
    logger.info("Top 3 relevant cases:")
    for case_id, score, title in top_relevant_cases:
        logger.info(f"Case ID: {case_id}, Title: {title}, Similarity Score: {score}")
 
    
    # Return top 3 cases with their similarity scores
    return [(case_id, score) for case_id, score, title in top_relevant_cases]

def prune_irrelevant_cases(cases, relevant_case_ids, max_cases=MAX_STORED_CASES):
    """Keep only the most relevant cases and limit total number stored"""
    # First, ensure all relevant cases are kept
    relevant_ids = [id for id, _ in relevant_case_ids]
    relevant_cases = [case for case in cases if case["id"] in relevant_ids]
    
    # Then add other cases until we reach the max
    other_cases = [case for case in cases if case["id"] not in relevant_ids]
    
    # Sort other cases by recency (assuming newer cases have higher IDs)
    other_cases.sort(key=lambda x: x["id"], reverse=True)
    
    # Combine and limit
    pruned_cases = relevant_cases + other_cases
    if len(pruned_cases) > max_cases:
        pruned_cases = pruned_cases[:max_cases]
        
    return pruned_cases

def needs_api_call_with_response(query, summary_context):
    """Determine if new case research is needed and optionally generate a response"""
    query = query.strip().lower()
    
    # Handle non-legal/short queries first
    if should_skip_api_call(query):
        return handle_non_legal_query(query)
    
    # Initial query without context always needs research
    if not summary_context:
        return {"isNewResearchRequired": True}
    
    # Determine if legal research needed based on context
    return evaluate_legal_research_need(query, summary_context)

# Helper functions below

def should_skip_api_call(query: str) -> bool:
    """Check if query is too short or a common greeting"""
    common_greetings = ["hi", "hello", "hey", "greetings", 
                       "good morning", "good afternoon", 
                       "good evening", "thanks", "thank you"]
    
    is_short = len(query) < 10
    is_greeting = any(greeting in query for greeting in common_greetings) and len(query) < 20
    return is_short or is_greeting

def handle_non_legal_query(query: str) -> dict:
    """Generate responses for non-legal queries"""
    prompt: str = ""
    
    if any(greeting in query for greeting in ["hi", "hello", "hey"]):
        logger.info(f"the greeting prompt is .. {_greeting_prompt(query)}")
        prompt = _greeting_prompt(query)
    else:
        logger.info(f"the short prompt is .. {_greeting_prompt(query)}")

        prompt = _short_query_prompt(query)
    
    response = gemini.generate_content(prompt)
    return _parse_response(response.text)



def evaluate_legal_research_need(query: str, context: str) -> dict:
    """Determine if legal research API call is needed"""
    research_prompt = legal_research_prompt(context, query)
    logger.info(f"the research prompt is .. {research_prompt}")

    response = gemini.generate_content(research_prompt)
    return _parse_response(response.text)

def _parse_response(response_text: str) -> dict:
    """Safely parse LLM response with fallbacks"""
    try:
        sanitized = sanitize_to_raw_json(response_text.strip())
        return json.loads(sanitized)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse response: {response_text}")
        return {
            "isNewResearchRequired": False,
            "responseToUser": "Please rephrase your legal query for better analysis."
        }
    
    
def update_summary_context(previous_context, query, response=None, new_cases=None):
    """Generate updated legal context summary with conversation history and case tracking"""
    # Build components using helper methods
    components = [
        _build_previous_context_section(previous_context),
        _build_current_query_section(query),
        _build_response_section(response),
        _build_new_cases_section(new_cases)
    ]
    
    # Construct full prompt from template
    template = _summary_prompt_template()  # Explicitly reference the function
    filtered_components = [c for c in components if c]
    
    full_prompt = template.format(
        components="\n\n".join(filtered_components),
        case_count=len(new_cases) if new_cases else 0
    )
    logger.info(f"the full prompt is .. {full_prompt}")

    
    # Generate and validate summary
    summary = _generate_and_validate_summary(full_prompt)
    return summary

# Helper functions ------------------------------------------------------------


def _build_previous_context_section(context):
    if not context:
        return "Initial Context: New legal analysis session"
    return f"Previous Legal Context:\n{context}"

def _build_current_query_section(query):
    return f"Current Legal Query:\n{query}"

def _build_response_section(response):
    if not response:
        return ""
    return f"Previous Analysis Summary:\n{_truncate_response(response)}"

def _build_new_cases_section(cases):
    if not cases:
        return ""
    
    return "New Case References:\n" + "\n".join(
        f"{idx+1}. {c.get('title', 'Untitled Case')} "
        f"({c.get('year', 'Year unknown')}): "
        f"{c.get('key_holding', 'No holding available')}"
        for idx, c in enumerate(cases[:3])
    )


def _truncate_response(response, max_words=75):
    words = response.split()[:max_words]
    return ' '.join(words) + ('...' if len(words) == max_words else '')

def _generate_and_validate_summary(prompt):
    response = gemini.generate_content(prompt)
    summary = response.text.strip()
    
    # Validation checks
    if len(summary) > 1000:
        summary = summary[:997] + "..."
    if not any(char in summary for char in [':', ';', '.']):
        summary = _convert_to_structured_summary(summary)
    
    return summary

def _convert_to_structured_summary(text):
    """Fallback structure for poorly formatted summaries"""
    sentences = text.split('.')[:3]
    return "\n".join(
        f"{idx+1}. {s.strip()}" 
        for idx, s in enumerate(sentences) 
        if s.strip()
    )


def get_historical_conversation(chat_id):
    try:
        return convex.query(
            "messages:historicalUserMessages",
            {"chatId": chat_id}  # Changed from "id" to "chatId"
        ) or []
    except Exception as e:
        logger.info(f"Error fetching conversation: {str(e)}")
        return []

def handle_query(chat_id, query):
    """Main function to handle a legal query with optimized storage"""
    # Fetch context from Convex

    logger.info(f"the chat id is .. {chat_id}")
    ctx = convex.query("chats:get", {"id": chat_id}) or {}

    summary_context = ctx.get("summary_context", "")
    retrieved_cases = ctx.get("retrieved_cases", "")
    conversation = ctx.get("conversation", [])
    
    # Check if API call needed and potentially get immediate response
    research_result = needs_api_call_with_response(query, summary_context)
    logger.info(f"the research result is .. {research_result}")
    
    # If we already have a response, return it directly
    if not research_result.get("isNewResearchRequired", True) and "responseToUser" in research_result:
        answer = research_result["responseToUser"]
        
        # Update conversation
        conversation.append({"role": "user", "content": query})
        conversation.append({"role": "assistant", "content": answer})
        
        
        
        return answer
    
    # Check if API call needed
    optimized_query = None
    new_cases = []
    if research_result.get("isNewResearchRequired", True):
        # Generate optimized search query
        optimized_query = prepare_search_query(query, summary_context)
        new_cases = search_indian_kanoon(optimized_query, query)
        
        #if new_cases:
            # Add new cases to retrieved_cases
        #    retrieved_cases.extend(new_cases)
            
            # We'll update the summary context AFTER generating the response
    
    # Find relevant cases from all retrieved cases
    #relevant_case_ids = get_relevant_cases(query, new_cases)
    
    # Prune irrelevant cases to keep storage manageable
    #retrieved_cases = prune_irrelevant_cases(retrieved_cases, relevant_case_ids)
    
    # Get the actual case objects for relevant cases
    #relevant_cases = [case for case in new_cases if case["id"] in [id for id, _ in relevant_case_ids]]
    
    # Prepare relevant case text for context
    case_text = ""

    
    case_titles = set()
    
    for case in new_cases:
        case_text += f"CASE: {case['title']} ({case['court']}, {case['year']})\n"
        case_text += f"CASE DETAILS: {case['doc_text']}\n\n"
        case_titles.add(case['title'])

    
    historical_conversartion = get_historical_conversation(chat_id);
    logger.info(f"calling big AI")
    # Generate response
    prompt = get_final_legal_query_resolver_prompt(summary_context, case_text, query, historical_conversartion)
    
    full_ai_message =  generate_ai_legal_full_prompt(prompt)

    response = openRouterResponse(full_ai_message)

    answer = response
    
    # Convert case_titles set to a string
    case_titles_str = "; ".join(case_titles)

    # NOW update the summary context with the query, response, and any new cases
    summary_context = update_summary_context(
        previous_context=summary_context,
        query=query,
        response=answer,  # Include the assistant's response
        new_cases=new_cases if new_cases else None
    )
    
    
    logger.info(f"the chat id is .. {chat_id}")

    
    convex.mutation("chats:updateChat", {
        "id": chat_id,
        "summary_context": summary_context,
        "retrieved_cases": case_titles_str,
    })
    

    return answer


def generate_ai_legal_full_prompt(prompt):
    system_message = {
        "role": "system", 
        "content": get_system_prompt2()
    }
    
    user_message = {
        "role": "user",
        "content": prompt
    }
    
    messages = [system_message, user_message]
    return messages

def generate_search_query_full_prompt(user_query, summary_context):
    system_message = {
        "role": "system", 
        "content": get_search_prompt_system()
    }
    
    user_message = {
        "role": "user",
        "content": get_search_prompt_user(user_query, summary_context)
    }
    
    messages = [system_message, user_message]
    return messages




def openRouterResponse(full_ai_message): 
    
    # Get OpenRouter response
    response = openrouter.generate_response(
        messages=full_ai_message,
        temperature=0.2,
        max_tokens=10000
    )
    
    if response and 'choices' in response:
        answer = response['choices'][0]['message']['content'].strip()
    else:
        # Fallback to Gemini if OpenRouter fails
        logger.error("OpenRouter failed, falling back to Gemini")
        response = gemini.generate_content(full_ai_message)
        answer = response.text.strip()
    
    return answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indian Legal Assistant")
    parser.add_argument("--chat-id", help="Existing chat session ID")
    parser.add_argument("--query", help="Legal query")
    args = parser.parse_args()
    
    if not args.query:
        parser.error("the --query argument is required")
    
    chat_id = args.chat_id or create_chat_session()
    response = handle_query(chat_id, args.query)
    
    logger.info(f"\n[Chat ID: {chat_id}]")
    logger.info(f"Response: {response}\n")