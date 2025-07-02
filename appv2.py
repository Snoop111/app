# app.py
import os
try:
    import streamlit as st
except Exception:  # pragma: no cover - allow running without streamlit
    class _Dummy:
        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return None

            return _noop

    st = _Dummy()
try:
    import boto3
except Exception:  # pragma: no cover - allow running tests without boto3
    boto3 = None  # type: ignore
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
try:
    from botocore.exceptions import ClientError, EndpointConnectionError
except Exception:  # pragma: no cover - allow running tests without botocore
    class ClientError(Exception):
        pass

    class EndpointConnectionError(Exception):
        pass
from dataclasses import dataclass

# Load environment variables from a `.env` file if present
def load_env() -> None:
    """Simple .env loader to avoid external dependencies."""
    if not os.path.exists('.env'):
        return
    with open('.env') as env_file:
        for line in env_file:
            if line.strip() and not line.strip().startswith('#'):
                key, _, value = line.strip().partition('=')
                if key and value:
                    os.environ.setdefault(key, value)

load_env()

# Configure logging
def setup_logging() -> logging.Logger:
    """Configure logging based on the LOG_LEVEL env var."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# --- Configuration ---
@dataclass
class AppConfig:
    """Application configuration settings loaded from environment."""

    REGION: str = os.getenv("REGION", "us-east-1")
    MODEL_ID: str = os.getenv("MODEL_ID", "claude-3-7-sonnet-20250219")
    EMBEDDING_MODEL_ID: str = os.getenv(
        "EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
    )
    KB_ID: str = os.getenv("KB_ID", "atlas_kb")
    RAG_MODEL_ID: str = os.getenv("RAG_MODEL_ID", MODEL_ID)
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", 500))
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", 10))

# Create config instance
config = AppConfig()

# --- Config Validation ---
def validate_config() -> None:
    """
    Validate required configuration parameters.
    
    Raises:
        SystemExit: If any required config is missing or contains placeholder values
    """
    errors = []
    config_dict = {
        "REGION": config.REGION,
        "MODEL_ID": config.MODEL_ID,
        "EMBEDDING_MODEL_ID": config.EMBEDDING_MODEL_ID,
        "KB_ID": config.KB_ID,
        "RAG_MODEL_ID": config.RAG_MODEL_ID,
        "MAX_TOKENS": config.MAX_TOKENS,
        "RATE_LIMIT_PER_MINUTE": config.RATE_LIMIT_PER_MINUTE,
    }
    
    for name, value in config_dict.items():
        if not value or (isinstance(value, str) and "your-" in value):
            errors.append(f"❌ {name} is not configured (current: '{value}')")
    
    if errors:
        st.error("## Configuration Errors")
        for error in errors:
            st.error(error)
        logger.error(f"Configuration validation failed: {errors}")
        st.stop()

validate_config()

# --- AWS Clients ---
def create_aws_clients() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Create and return AWS Bedrock clients.
    
    Returns:
        Tuple of (bedrock_runtime_client, bedrock_agent_runtime_client)
    """
    if boto3 is None:
        logger.warning("boto3 is not available; AWS clients are disabled")
        return None, None
    try:
        bedrock = boto3.client("bedrock-runtime", region_name=config.REGION)
        bedrock_kb = boto3.client(
            "bedrock-agent-runtime", region_name=config.REGION
        )
        logger.info(f"AWS clients created in region {config.REGION}")
        return bedrock, bedrock_kb
    except Exception as e:
        logger.error(f"Failed to create AWS clients: {str(e)}")
        st.error(f"Failed to initialize AWS clients: {str(e)}")
        st.stop()

bedrock, bedrock_kb = create_aws_clients()

# --- Rate Limiting ---
class RateLimiter:
    """
    Rate limiter to control API call frequency.
    
    Attributes:
        max_calls (int): Maximum number of calls allowed per minute
        calls (List[float]): Timestamps of recent API calls
    """
    
    def __init__(self, max_calls_per_minute: int) -> None:
        """
        Initialize rate limiter.
        
        Args:
            max_calls_per_minute: Maximum API calls allowed per minute
        """
        self.max_calls = max_calls_per_minute
        self.calls: List[float] = []
    
    def check_rate_limit(self) -> None:
        """
        Check and enforce rate limiting.
        
        Blocks execution if rate limit is exceeded until calls are allowed again.
        """
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [call for call in self.calls if now - call < 60]
        
        if len(self.calls) >= self.max_calls:
            wait_time = 60 - (now - self.calls[0])
            logger.warning(f"Rate limit exceeded, waiting {wait_time:.1f} seconds")
            placeholder = st.empty()
            for remaining in range(int(wait_time), 0, -1):
                placeholder.warning(
                    f"Rate limit exceeded. Waiting {remaining}s..."
                )
                time.sleep(1)
            placeholder.empty()
            self.calls.pop(0)
        
        self.calls.append(now)

rate_limiter = RateLimiter(config.RATE_LIMIT_PER_MINUTE)

# --- Helper Functions ---
try:
    import tiktoken  # type: ignore
    ENCODER = tiktoken.encoding_for_model(config.MODEL_ID)
except Exception:  # pragma: no cover - fallback if tiktoken isn't installed
    ENCODER = None


def estimate_tokens(text: str) -> int:
    """Estimate token count using tiktoken when available."""
    if ENCODER:
        return len(ENCODER.encode(text))
    # Rough fallback: assume 1 token ≈ 4 characters
    return len(text) // 4

def validate_token_limit(text: str) -> bool:
    """
    Check if text is within token limits.
    
    Args:
        text: Text to validate
        
    Returns:
        True if within limits, False otherwise
    """
    return estimate_tokens(text) <= config.MAX_TOKENS

def truncate_response(text: str) -> str:
    """
    Truncate response text if it exceeds token limits.
    
    Args:
        text: Response text to potentially truncate
        
    Returns:
        Original text or truncated version with notice
    """
    if validate_token_limit(text):
        return text
    
    max_chars = config.MAX_TOKENS * 4
    truncated_text = text[:max_chars]
    return truncated_text + "\n\n[Response truncated due to token limits]"

def chat_with_kb(
    question: str, 
    kb_id: str = None, 
    model_id: str = None, 
    top_k: int = 3
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieve and generate response from Knowledge Base using RAG.
    
    Args:
        question: User's question/query
        kb_id: Knowledge base ID (defaults to config value)
        model_id: Model ID for RAG (defaults to config value)
        top_k: Number of top documents to retrieve
        
    Returns:
        Tuple of (generated_answer, source_documents)
        
    Raises:
        ValueError: If question exceeds token limits
    """
    # Use defaults from config if not provided
    kb_id = kb_id or config.KB_ID
    model_id = model_id or config.RAG_MODEL_ID
    
    try:
        # Validate input
        if not validate_token_limit(question):
            raise ValueError(
                f"Question exceeds maximum token limit ({config.MAX_TOKENS} tokens)"
            )
        
        # Check rate limits
        rate_limiter.check_rate_limit()
        
        # Build model ARN
        model_arn = (
            f"arn:aws:bedrock:{config.REGION}::foundation-model/{model_id}"
        )
        
        # Make API call
        response = bedrock_kb.retrieve_and_generate(
            input={'text': question},
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': kb_id,
                    'modelArn': model_arn,
                    'retrievalConfiguration': {
                        'vectorSearchConfiguration': {
                            'numberOfResults': top_k
                        }
                    }
                }
            }
        )
        
        # Extract response data
        answer = response['output']['text']
        docs = response.get('citations', [])
        
        # Truncate response if needed
        answer = truncate_response(answer)
        
        logger.info(f"Successfully generated response with {len(docs)} citations")
        return answer, docs
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        request_id = e.response.get('ResponseMetadata', {}).get('RequestId')
        
        if error_code == 'ThrottlingException':
            logger.warning(f"AWS throttling: {error_message}")
            st.warning("Request was throttled. Please try again in a moment.")
            return "I'm experiencing high demand. Please try again shortly.", []
        elif error_code == 'ValidationException':
            logger.error(f"Validation error: {error_message}")
            return f"Invalid request: {error_message}", []
        else:
            logger.error(
                f"AWS error ({error_code}): {error_message} [RequestId: {request_id}]"
            )
            return f"AWS service error: {error_message}", []

    except EndpointConnectionError as e:
        logger.error(f"Network error: {str(e)}")
        return "Network error communicating with AWS services.", []
            
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        st.error(str(e))
        return f"Error: {str(e)}", []
        
    except Exception as e:
        logger.error(f"Unexpected error in chat_with_kb: {str(e)}")
        st.error(f"Unexpected error: {str(e)}")
        return f"Sorry, I encountered an unexpected error: {str(e)}", []

# --- Streamlit UI ---
def setup_page_config() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Bedrock RAG Chat", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_docs" not in st.session_state:
        st.session_state.last_docs = []

def render_sidebar() -> None:
    """Render the sidebar with configuration and controls."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.caption(f"**Knowledge Base:** {config.KB_ID}")
        st.caption(f"**Model:** {config.MODEL_ID}")
        st.caption(f"**Max Tokens:** {config.MAX_TOKENS:,}")
        st.caption(f"**Rate Limit:** {config.RATE_LIMIT_PER_MINUTE}/min")
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_docs = []
            st.success("Chat history cleared!")

def render_chat_history() -> None:
    """Render the chat message history."""
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def render_source_documents() -> None:
    """Render source documents if available."""
    if st.session_state.last_docs:
        with st.expander("📄 Source Documents", expanded=False):
            for i, doc in enumerate(st.session_state.last_docs, 1):
                content = doc.get('content', {}).get('text', 'No content available')
                source_uri = (
                    doc.get('location', {})
                    .get('s3Location', {})
                    .get('uri', 'Unknown source')
                )
                
                st.markdown(f"**Document {i}**")
                st.text(content[:500] + "..." if len(content) > 500 else content)
                st.caption(f"**Source:** {source_uri}")
                
                if i < len(st.session_state.last_docs):
                    st.divider()

def main() -> None:
    """Main application function."""
    setup_page_config()
    initialize_session_state()
    
    st.title("💼 AWS Bedrock Knowledge Base Chat")
    st.markdown("Ask questions and get AI-powered answers from your knowledge base!")
    
    render_sidebar()
    
    # Main chat interface
    query = st.chat_input("Ask me anything about the knowledge base...")
    
    if query:
        # Add user message to history
        st.session_state.history.append({"role": "user", "content": query})
        
        # Display loading state with progress
        with st.status("🧠 Consulting knowledge base...", expanded=True) as status:
            st.write("🔍 Searching for relevant information...")
            start_time = time.time()
            
            # Get response from Bedrock
            answer, docs = chat_with_kb(query)
            
            # Store results
            st.session_state.last_docs = docs
            st.session_state.history.append({"role": "assistant", "content": answer})
            
            # Show performance metrics
            latency = time.time() - start_time
            st.write(f"✅ Response generated in {latency:.2f}s")
            st.write(f"📊 Estimated tokens: {estimate_tokens(answer):,}")
            st.write(f"📚 Sources found: {len(docs)}")
            
            status.update(label="✅ Response ready!", state="complete")
    
    # Display chat history and sources
    render_chat_history()
    render_source_documents()

if __name__ == "__main__":
    main()
