import logging
from src.rag.prompts import RAG_ASSISTANT_PROMPT
from src.rag.llm_provider import get_llm_provider
from config import OLLAMA_LLM_MODEL, OLLAMA_BASE_URL, VLLM_BASE_URL, DEBUG_MODE, LLM_PROVIDER

logger = logging.getLogger(__name__)

# Map provider type → its default base URL
_PROVIDER_DEFAULT_URL = {
    "ollama": OLLAMA_BASE_URL,
    "vllm":   VLLM_BASE_URL,
}


class ReasoningGenerator:
    def __init__(self, model_name: str = None, provider_type: str = None, base_url: str = None):
        """
        Initialize ReasoningGenerator with configurable LLM provider.

        Args:
            model_name:    Name of the model (default: from config)
            provider_type: "ollama" | "vllm" | "hf" | "mock"
            base_url:      Override the server URL (auto-detected from provider_type if None)
        """
        self.model_name = model_name or OLLAMA_LLM_MODEL
        self.provider_type = provider_type or LLM_PROVIDER

        # Auto-pick the right default URL for the chosen provider
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = _PROVIDER_DEFAULT_URL.get(self.provider_type, OLLAMA_BASE_URL)

        # Initialize the LLM provider
        self.provider = get_llm_provider(
            model_name=self.model_name,
            provider_type=self.provider_type,
            base_url=self.base_url,
        )
        
        if DEBUG_MODE:
            logger.info(f"ReasoningGenerator initialized with model: {self.model_name}, provider: {self.provider_type}")

    def generate_answer(self, query: str, context_docs: list, temperature: float = 0.7) -> str:
        """
        Generates a grounded clinical answer using retrieved context.

        Args:
            query (str): The user's clinical question.
            context_docs (list): Retrieved LangChain Document objects.
            temperature (float): Model temperature for response generation.

        Returns:
            str: The model's grounded answer.
        """
        # Combine retrieved chunks into a single context string
        context_text = "\n\n---\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')} | Section: {doc.metadata.get('section', 'General')}]\n{doc.page_content}"
            for doc in context_docs
        ])

        # Separate system prompt from user message for better model compliance
        messages = [
            {
                "role": "system",
                "content": RAG_ASSISTANT_PROMPT
            },
            {
                "role": "user",
                "content": (
                    f"CONTEXT FROM CLINICAL GUIDELINES:\n"
                    f"{context_text}\n\n"
                    f"QUESTION:\n{query}\n\n"
                    f"ANSWER:"
                )
            }
        ]

        try:
            response = self.provider.generate(
                messages=messages,
                temperature=temperature
            )
            return response

        except Exception as e:
            logger.error(f"Error in answer generation: {str(e)}")
            return f"Error in generation: {str(e)}"