import os
import json
import argparse
import requests
from convex import ConvexClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

# Load environment file from the same package
load_dotenv()


# Initialize clients - updated Convex initialization
convex_url = os.getenv("CONVEX_URL")
print(f"convex url is: {convex_url}")

convex = ConvexClient(convex_url)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
generation_config = {
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 500,
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
MAX_SEARCH_RESULTS_FROM_KANOON = 5

def create_chat_session():
    """Create a new chat session and return its ID"""
    chat_id = f"chat_{os.urandom(4).hex()}"
    convex.mutation("chats:insert", {
        "chat_id": chat_id,
        "summary_context": "",
        "retrieved_cases": [],
        "conversation": []
    })
    return chat_id

def generate_case_summary(query, full_text, title, max_length=MAX_SUMMARY_LENGTH):
    """Generate a concise summary of a legal case using Gemini"""
    prompt = f"""
    Summarize the following legal case in 2-3 short sentences (maximum {max_length} characters) to addresst the query being asked.

    QUERY: {query}
    
    TITLE: {title}
    
    CONTENT: {full_text}  # Limit input to avoid token limits
    
    Provide only the summary text with no additional commentary.
    """
    
    try:
        response = gemini.generate_content(prompt)
        summary = response.text.strip()
        # Ensure summary doesn't exceed max length
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        # Fallback to simple truncation
        return full_text[:max_length-3] + "..."
    


def prepare_search_query(user_query, summary_context):
    """Generate optimized search parameters for Indian Kanoon based on user query and context"""
    prompt = f"""
    Task: Generate an optimized search query for Indian legal case database.
    
    USER QUERY: {user_query}
    
    CONVERSATION CONTEXT: {summary_context}
    
    Based on the user's query and our conversation context, construct the most effective search query for finding relevant Indian legal cases. 
    Extract key legal terms, statutes, or case references.
    Focus on specific legal questions rather than general conversation context.
    
    Return ONLY the search query text as a single string, no explanation or additional text.
    """
    
    try:
        response = gemini.generate_content(prompt)
        search_query = response.text.strip()
        print(f"Generated search query: {search_query}")
        return search_query
    except Exception as e:
        print(f"Error generating search query: {e}")
        # Fallback to original query if there's an error
        return user_query

def search_indian_kanoon(query, summary_context, pagenum=0, max_results=MAX_SEARCH_RESULTS_FROM_KANOON):
    """Search Indian Kanoon API for relevant cases with optimized storage"""
    # Generate optimized search query
    optimized_query = prepare_search_query(query, summary_context)

    #remove double quotes from start and end if they exist
    optimized_query = optimized_query[1:-1]
    
    print(f"optimized query to be sent to search : {optimized_query}")

    headers = {
        "Authorization": f"Token {INDIAN_KANOON_API_KEY}"
    }
    params = {
        "formInput": optimized_query,  # Use the optimized query instead of raw user query
        "pagenum": pagenum,
        "maxcites": 3  # Get citations with results
    }
    
    print(f"the params to search are : {params}")
    response = requests.post(INDIAN_KANOON_SEARCH_URL, headers=headers, params=params)
    if response.status_code != 200:
        print(f"Error calling Indian Kanoon API: {response.status_code}")
        return []
    
    results = response.json()
    print(f"the response from Indian Kanoon search is .. {results}")
    cases = []
    
    # Process up to max_results cases
    for doc in results.get("docs", [])[:max_results]:
        doc_id = doc.get("tid")
        if not doc_id:
            continue
            
        print(f"Fetching document from Indian Kanoon for doc id: {doc_id}")
        # Get document metadata
        meta_response = requests.post(
            f"{INDIAN_KANOON_DOC_URL}{doc_id}/", 
            headers=headers
        )
        
        if meta_response.status_code == 200:
            meta = meta_response.json()
            
            # Extract document text
            doc_text = meta.get("doc", "")
            
            # Generate a concise summary instead of storing full text
            title = meta.get("title", "Untitled Case")
            summary = generate_case_summary(query, doc_text, title)
            
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
                "id": str(doc_id),
                "title": title,
                "summary": summary,  # This is now a concise summary
                "court": meta.get("court", "Unknown Court"),
                "year": year,
                "citations": [str(citation) for citation in citations][:3],  # Limit citations stored
                "doc_url": f"{INDIAN_KANOON_DOC_URL}{doc_id}/"
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
    case_embeds = {c["id"]: model.encode(c["summary"]) for c in cases}
    similarities = {cid: cosine_similarity(query_embed, e) for cid, e in case_embeds.items()}
    
    # Sort cases by similarity
    sorted_case_ids = sorted(similarities.items(), key=lambda x: -x[1])
    
    # Return top 3 cases with their similarity scores
    return [(case_id, score) for case_id, score in sorted_case_ids[:3]]

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
    # For very short queries or common greetings, don't call the API
    query = query.strip().lower()
    common_greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "thanks", "thank you"]
    
    # Short circuit for obvious non-legal queries with direct response
    if len(query) < 10 or (any(greeting in query for greeting in common_greetings) and len(query) < 20):
        print("Query detected as greeting or too short - skipping API call")
        
        # Generate appropriate response for non-legal query
        if any(greeting in query for greeting in common_greetings):
            prompt = f"""
            The user has sent what appears to be a greeting: "{query}"
            
            Respond with a friendly greeting as a legal assistant and ask how you can help with their legal query.
            Structure your response in JSON format:
            {{
                "isNewResearchRequired": false,
                "responseToUser": "your greeting response here"
            }}
            """
        else:
            prompt = f"""
            The user has sent a very brief query: "{query}"
            
            Respond politely and ask for more details about their legal query.
            Structure your response in JSON format:
            {{
                "isNewResearchRequired": false,
                "responseToUser": "your response here asking for more details"
            }}
            """
            
        response = gemini.generate_content(prompt)
        try:
            result = json.loads(response.text.strip())
            return result
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "isNewResearchRequired": False,
                "responseToUser": "Hello! I'm your Indian legal assistant. How can I help with your legal query today?"
            }
        
    if not summary_context:
        return {"isNewResearchRequired": True}  # Always call API for first substantive query
        
    prompt = f"""
    Task: Determine if this user query requires legal research and provide an appropriate response.
    
    First, determine if this is a legal query that requires access to legal cases or if it's casual conversation.
    If it's just casual conversation or a non-legal question, generate an appropriate response.
    
    Existing Context Summary:
    {summary_context}
    
    User Query:
    {query}
    
    Respond with a JSON object with the following structure:
    {{
        "isNewResearchRequired": true/false,
        "responseToUser": "only include this if no research is needed"
    }}
    
    If this is a legal query requiring new research, set isNewResearchRequired to true and don't include responseToUser.
    If this is NOT a legal query OR can be answered from existing context, set isNewResearchRequired to false and include an appropriate responseToUser.
    """
    
    response = gemini.generate_content(prompt)
    result_text = response.text.strip()
    
    try:
        # Try to parse the response as JSON

        print(f"the response is .. {result_text}")
        result = json.loads(result_text)
        print(f"Research decision: {result}")
        return result
    except json.JSONDecodeError:
        # If JSON parsing fails, check for explicit mentions of research needed
        if "true" in result_text.lower():
            return {"isNewResearchRequired": True}
        else:
            # Generate a fallback response
            return {
                "isNewResearchRequired": False,
                "responseToUser": "I understand your question. Based on our conversation so far, I can provide some insights without additional research."
            }

def update_summary_context(previous_context, query, response=None, new_cases=None):
    """Generate an updated summary context that includes the assistant's response. """
    # Base prompt with previous context and current query
    base_prompt = f"""
    Task: Update the legal context summary
    
    Previous Context Summary:
    {previous_context}
    
    Current Query:
    {query}
    """
    
    # Add response information if available
    if response:
        base_prompt += f"""
    Assistant's Response Summary:
    {response}  # Limit to avoid token limits
    """
    
    # Add new cases if available
    if new_cases:
        case_summaries = "\n".join([f"- {c['title']}: {c['summary']}" for c in new_cases[:3]])
        base_prompt += f"""
    New Cases:
    {case_summaries}
    """
    
    # Complete the prompt
    base_prompt += """
    Provide a concise sentence summary that combines all the above information into a coherent context summary for future reference.
    Include the cases that have been cited in the assistant's response summary so that we don't end up citing the same cases.
    Limit your response to 600 characters maximum.
    """
    
    response = gemini.generate_content(base_prompt)
    new_summary = response.text.strip()
    
    # Ensure summary doesn't exceed reasonable length
    if len(new_summary) > 1000:
        new_summary = new_summary[:997] + "..."
        
    return new_summary

def handle_query(chat_id, query):
    """Main function to handle a legal query with optimized storage"""
    # Fetch context from Convex
    ctx = convex.query("chats:get", {"chat_id": chat_id}) or {}
    summary_context = ctx.get("summary_context", "")
    retrieved_cases = ctx.get("retrieved_cases", [])
    conversation = ctx.get("conversation", [])
    
    # Check if API call needed and potentially get immediate response
    research_result = needs_api_call_with_response(query, summary_context)
    
    # If we already have a response, return it directly
    if not research_result.get("isNewResearchRequired", True) and "responseToUser" in research_result:
        answer = research_result["responseToUser"]
        
        # Update conversation
        conversation.append({"role": "user", "content": query})
        conversation.append({"role": "assistant", "content": answer})
        
        # Keep only the most recent 20 messages (10 exchanges)
        if len(conversation) > 20:
            conversation = conversation[-20:]
        
        # Update Convex with conversation only (no new cases or context changes)
        convex.mutation("chats:update", {
            "chat_id": chat_id,
            "conversation": conversation
        })
        
        return answer
    
    # Check if API call needed
    new_cases = []
    if research_result.get("isNewResearchRequired", True):
        new_cases = search_indian_kanoon(query, summary_context)
        
        if new_cases:
            # Add new cases to retrieved_cases
            retrieved_cases.extend(new_cases)
            
            # We'll update the summary context AFTER generating the response
    
    # Find relevant cases from all retrieved cases
    relevant_case_ids = get_relevant_cases(query, retrieved_cases)
    
    # Prune irrelevant cases to keep storage manageable
    retrieved_cases = prune_irrelevant_cases(retrieved_cases, relevant_case_ids)
    
    # Get the actual case objects for relevant cases
    relevant_cases = [case for case in retrieved_cases if case["id"] in [id for id, _ in relevant_case_ids]]
    
    # Prepare relevant case text for context
    case_text = ""
    for case in relevant_cases:
        case_text += f"CASE: {case['title']} ({case['court']}, {case['year']})\n"
        case_text += f"SUMMARY: {case['summary']}\n\n"
    
    # Generate response
    prompt = f"""
    You are an AI legal assistant specializing in Indian law.
    
    CONTEXT SUMMARY: 
    {summary_context}
    
    RELEVANT LEGAL PRECEDENTS:
    {case_text}
    
    USER QUERY: 
    {query}
    
    Provide a comprehensive legal analysis addressing the query, citing relevant cases from the context. 
    Be specific with your references to cases and legal principles. 
    Format your answer clearly with appropriate sections and explain any legal terms for a non-expert.
    """
    
    response = gemini.generate_content(prompt)
    answer = response.text.strip()
    
    # NOW update the summary context with the query, response, and any new cases
    summary_context = update_summary_context(
        previous_context=summary_context,
        query=query,
        response=answer,  # Include the assistant's response
        new_cases=new_cases if new_cases else None
    )
    
    # Store only the last 10 conversation entries to prevent unlimited growth
    conversation = ctx.get("conversation", [])
    conversation.append({"role": "user", "content": query})
    conversation.append({"role": "assistant", "content": answer})
    
    # Keep only the most recent 20 messages (10 exchanges)
    if len(conversation) > 20:
        conversation = conversation[-20:]
    
    # Update Convex with optimized data
    convex.mutation("chats:update", {
        "chat_id": chat_id,
        "summary_context": summary_context,
        "retrieved_cases": retrieved_cases,
        "conversation": conversation
    })
    
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
    
    print(f"\n[Chat ID: {chat_id}]")
    print(f"Response: {response}\n")