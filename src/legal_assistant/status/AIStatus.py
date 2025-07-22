from enum import Enum, auto

class AIStatus(Enum):
    # Initial States
    RECEIVED = "Request received and queued"
    VALIDATING = "Validating request parameters"
    
    # Processing States
    INITIALIZING = "Recieved and Initializing"
    SEARCHING_DATABASES = "Running Vectors"
    ANALYZING_CASES = "Comparing cosine scores"
    #PROCESSING_STATUTES = "Processing statutes and regulations"
    CROSS_REFERENCING = "Cross-referencing"
    SYNTHESIZING = "Synthesizing research findings"
    
    # Final States
    COMPLETED = "Research completed successfully"
    ERROR = "An error occurred during processing"
    TIMEOUT = "Request timed out"
    CANCELLED = "Request was cancelled by user"
