
import os
from typing import Any, List, Optional, Iterator
from dotenv import load_dotenv
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from byteplussdkarkruntime import Ark

# Load environment variables
load_dotenv()

class BytePlusLLM(LLM):
    """
    Custom LLM Wrapper for BytePlus Ark.
    """
    api_key: str
    endpoint_id: str
    client: Any = None

    def __init__(self, api_key: Optional[str] = None, endpoint_id: Optional[str] = None, **kwargs):
        api_key = api_key or os.getenv("ARK_API_KEY")
        endpoint_id = endpoint_id or os.getenv("MODEL_ENDPOINT_ID")
        
        if not api_key or not endpoint_id:
            raise ValueError("ARK_API_KEY and MODEL_ENDPOINT_ID must be set in environment or passed directly.")
            
        super().__init__(api_key=api_key, endpoint_id=endpoint_id, **kwargs)
        self.client = Ark(
            api_key=self.api_key,
            base_url="https://ark.ap-southeast.bytepluses.com/api/v3"
        )

    @property
    def _llm_type(self) -> str:
        return "byteplus_ark"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant for ScholarSync logic."},
                {"role": "user", "content": prompt}
            ]
            
            completion = self.client.chat.completions.create(
                model=self.endpoint_id,
                messages=messages,
                stop=stop # Pass stop sequences to the API
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error calling BytePlus API: {e}"

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """
        Stream the response from BytePlus Ark.
        """
        try:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant for ScholarSync logic."},
                {"role": "user", "content": prompt}
            ]

            stream = self.client.chat.completions.create(
                model=self.endpoint_id,
                messages=messages,
                stop=stop,
                stream=True  # Enable streaming
            )

            for chunk in stream:
                # BytePlus/OpenAI chunks usually put content in choices[0].delta.content
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content is not None:
                        token = delta.content
                        yield GenerationChunk(text=token)
                        if run_manager:
                            run_manager.on_llm_new_token(token)

        except Exception as e:
            # Yield error as text so it doesn't crash the stream silently
            yield GenerationChunk(text=f"Error streaming BytePlus API: {e}")

def get_llm():
    """
    Returns the configured LLM instance.
    You can switch this to ChatOpenAI if you want to use OpenAI instead.
    """
    # For now, strictly use the requested BytePlus setup if keys exist, 
    # otherwise fallback or error would be handled by the class.
    return BytePlusLLM()
