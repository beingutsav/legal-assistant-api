def get_search_query_prompt(user_query, summary_context):
    return f"""
            Task: Generate a Google search query optimized for retrieving highly specific Indian legal case documents from Indian Kanoon. The query will be executed via Google and results fed to lawyers for case research.

        Technical Requirements:
        1. Use strict Boolean logic with AND/OR operators
        2. Mandatory quotes for exact phrases ("anticipatory bail")
        3. Prioritize specificity over recall - aim for 50-100 perfect matches rather than 1,000 vague ones
        4. Exclude site: operators (will be auto-appended)
        5. Format as single-line text without markdown

        Construction Rules:
        - MUST include:
        a) Legal remedy ("anticipatory bail", "writ petition")
        b) Offense/statute ("murder", "Section 302 IPC")
        c) Outcome keywords ("granted", "quashed", "dismissed")
        d) Landmark case references when contextually relevant (e.g., "Arnesh Kumar vs State of Bihar")
        - MUST exclude:
        a) Non-legal terms
        b) Procedural phrases ("how to", "can I")
        c) Broad modifiers ("recent", "important")

        Example Pattern:
        ("anticipatory bail" OR "pre-arrest bail") AND ("murder" OR "IPC 302") AND ("granted" OR "denied") -"quash" -"dismissed"

        User Query: {user_query}

        Conversation Context: {summary_context}

        Generate query by:
        1. Extracting core legal elements from query/context
        2. Identifying synonymous legal terms (e.g., "CrPC 438" = "anticipatory bail")
        3. Adding exclusion terms (-"irrelevant_term") to filter noise
        4. Combining using Boolean logic for maximal precision
    """


def get_final_legal_query_resolver_prompt(summary_context, case_text, query, historical_queries):
        # Define reusable components
        base_context = {
            "role": "Senior Advocate, Supreme Court of India",
            "experience": "40+ years in constitutional/criminal law",
            "audience": "Seasoned litigation lawyers",
            "depth_requirement": "Nuanced interpretation needed"
        }

        response_rules = f"""
        **Response Protocol**
        1. Cross-reference 3-5 key cases from KNOWLEDGE BASE when relevant
        2. Analyze conflicting precedents using {base_context['experience']}
        3. Highlight overlooked aspects of cited judgments
        4. Use general knowledge ONLY if KNOWLEDGE BASE is empty
        5. Provide response in markdown format as it would be shown on a chat interface
        """

        conversation_policy = f"""
        **Conversation Management**
        - Previous User queries: {historical_queries or 'None'}
        - Current context: {summary_context or 'New case analysis'}
        - Handling: {"Continue existing analysis" if summary_context else "Establish new framework"}
        """

        return f"""
        ## Legal Strategy Directive (v2.1)
        **Role**: {base_context['role']} advising {base_context['audience']}

        ### Core Requirements
        {response_rules}

        ### Contextual Parameters
        {conversation_policy}

        ### Mandatory Structure
        ### [Subject:
        **Doctrinal Analysis**
        1. Current position (cite 2-3 {base_context['role'].split()[-1]} cases)
        2. Judicial conflicts
        3. Post-2020 trends

        **Strategic Map**
        - Petition drafting: Emphasize sections matching {case_text[:25]}...
        - Forum selection: Prioritize benches with favorable precedents
        ```

        ### Input Modules
        KNOWLEDGE BASE: {case_text or 'No cases provided'}
        HISTORICAL QUERIES: {historical_queries or 'None'}
        LIVE QUERY: {query}
    """

def _greeting_prompt(query):
    return f"""
        Generate a friendly legal assistant greeting response to: "{query}"
        Respond in JSON format only: {{
            "isNewResearchRequired": false,
            "responseToUser": "<appropriate greeting>"
        }}"""
    

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




def _short_query_prompt(query: str) -> str:
    return f"""
        Politely ask for more legal details about: "{query}"
        Respond in JSON format only: {{
            "isNewResearchRequired": false,
            "responseToUser": "<request for clarification>"
        }}"""
    

def legal_research_prompt(context, query) -> dict:
    return f'''
        Analyze if this legal query requires fresh research or can be answered from existing context:
        
        Context: {context}
        Query: {query}
        
        Respond ONLY with valid JSON: {{
            "isNewResearchRequired": true|false,
            "responseToUser": "<only if no research needed>"
        }}
        
        Decision Criteria:
        1. Needs research if query contains new legal questions not in context
        2. Needs research if asking for case references/statutes
        3. No research needed if asking for explanation of previous analysis'''




def _doc_summary_prompt(max_length, query, title, full_text) -> dict:
    return f"""
        Summarize the following legal case in 2-3 short sentences (maximum {max_length} characters) to addresst the query being asked.

        QUERY: {query}
        
        TITLE: {title}
        
        CONTENT: {full_text}  # Limit input to avoid token limits
        
        Provide only the summary text with no additional commentary.
    """