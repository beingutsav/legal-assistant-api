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
    full_prompt = _summary_prompt_template().format(
        components="\n\n".join([c for c in components if c]),
        case_count=len(new_cases) if new_cases else 0
    )
    
    # Generate and validate summary
    summary = _generate_and_validate_summary(full_prompt)
    return summary

# Helper functions ------------------------------------------------------------

def _summary_prompt_template():
    return """Legal Context Update Protocol

{components}

Synthesis Requirements:
1. Maintain chronological flow of legal analysis
2. Track cited cases: [List case titles from previous context and new cases]
3. Preserve key legal arguments from both parties
4. Highlight conflicting precedents if any
5. Note pending legal questions

Generate a 3-sentence summary (max 500 chars) for future reference:
- Sentence 1: Core legal issue + progression
- Sentence 2: Key cited cases/precedents
- Sentence 3: Outstanding questions/next steps"""

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
        f"{idx+1}. {c['title']} ({c['year']}): {c['key_holding']}" 
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