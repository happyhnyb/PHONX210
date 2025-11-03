import asyncio
import os
import json
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import logging
from ai_call_logger import CallLogger
from datetime import datetime

logging.basicConfig(level=logging.INFO)

load_dotenv()

# Centralized logs directory within insurance-ai
_LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOGS_DIR, exist_ok=True)

_TURN_LOG = os.path.join(_LOGS_DIR, "ai_turns.jsonl")
_CHUNKS_LOG = os.path.join(_LOGS_DIR, "ai_chunks.log")
_SUCCESS_LOG = os.path.join(_LOGS_DIR, "success_logs_openai.txt")

_turn_lock = asyncio.Lock()
_chunk_lock = asyncio.Lock()
_success_lock = asyncio.Lock()


async def _append_turn_log(messages: List[Dict[str, str]], response: str, *, streaming: bool, model: str, user_phone_number: Optional[str] = None) -> None:
    async with _turn_lock:
        try:
            record = {
                "timestamp": asyncio.get_event_loop().time(),
                "model": model,
                "streaming": bool(streaming),
                "user_phone_number": user_phone_number,
                "messages": messages,
                "response": (response or ""),
                "response_chars": len(response or ""),
            }
            with open(_TURN_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            # best-effort only
            pass


async def _append_chunk_log(seq: int, chunk_text: str, *, model: str, user_phone_number: Optional[str] = None) -> None:
    async with _chunk_lock:
        try:
            safe_text = (chunk_text or "").replace("\n", " ").replace("\r", " ").strip()
            line = f"model={model} | user={user_phone_number or ''} | seq={seq} | {safe_text}\n"
            with open(_CHUNKS_LOG, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            # best-effort only
            pass

async def _append_success_log(*, messages: Optional[List[Dict[str, str]]], streaming: bool, model: str, user_phone_number: Optional[str] = None) -> None:
    async with _success_lock:
        try:
            ts = datetime.utcnow().isoformat() + "Z"
            # Build a compact row representation of the outbound messages
            row = ""
            try:
                if messages:
                    parts = []
                    for m in messages:
                        role = (m.get("role") or "").strip()
                        content = (m.get("content") or "")
                        safe_content = content.replace("\n", " ").replace("\r", " ").strip()
                        parts.append(f"{role}:{safe_content}")
                    row = " || ".join(parts)
                    if len(row) > 1000:
                        row = row[:1000] + "..."
            except Exception:
                row = ""

            # Only success entries are written to this file
            line = f"[{ts}] HTTP 200 | model={model} | user={user_phone_number or ''} | streaming={bool(streaming)} -> {row}\n"
            with open(_SUCCESS_LOG, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            # best-effort only
            pass

class AsyncChatCompletion:
    """
    Async OpenAI Chat Completion handler with object-oriented design
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-chat-latest"):
        """
        Initialize the AsyncChatCompletion client
        
        Args:
            api_key: OpenAI API key (defaults to AI_KEY environment variable)
            model: Model to use for completions
        """
        self.api_key = api_key or "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"
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def create_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        call_logger: CallLogger = None,
        user_phone_number: str = None,
        **kwargs
    ) -> Any:
        """
        Create an async chat completion
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters for the API call
            
        Returns:
            OpenAI completion response
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Log the start of AI completion if logger is provided
            call_logger.log_event(user_phone_number, f"🤖 Starting AI completion with model: {self.model}")
            call_logger.log_event(user_phone_number, f"🌡️ Temperature: {temperature}, Max tokens: {max_tokens}")
            
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            end_time = asyncio.get_event_loop().time()
            completion_time = end_time - start_time
            logging.info(f"Completion created in {completion_time:.2f} seconds")
            # Success log (HTTP 200 equivalent)
            await _append_success_log(messages=messages, streaming=False, model=self.model, user_phone_number=user_phone_number)
            
            # Log the completion success if logger is provided
            
            call_logger.log_event(user_phone_number, f"✅ AI completion completed in {completion_time:.3f}s")
            if completion.choices and completion.choices[0].message:
                response_length = len(completion.choices[0].message.content)
                call_logger.log_event(user_phone_number, f"📝 Response generated: {response_length} chars")
            
            return completion
        except Exception as e:
            completion_time = asyncio.get_event_loop().time() - start_time if 'start_time' in locals() else 0
            logging.error(f"Error creating completion: {e}")
            
            # Log the error if logger is provided
            call_logger.log_event(user_phone_number, f"❌ AI completion failed after {completion_time:.3f}s: {str(e)[:100]}")
            
            raise
    
    async def get_response_content(
        self, 
        messages: List[Dict[str, str]], 
        call_logger: CallLogger = None,
        user_phone_number: str = None,
        **kwargs
    ) -> str:
        """
        Get just the content of the response message
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters for create_completion
            
        Returns:
            The content string from the first choice
        """
        completion = await self.create_completion(
            messages, 
            call_logger=call_logger,
            user_phone_number=user_phone_number,
            **kwargs
        )
        content = completion.choices[0].message.content
        # Best-effort centralized turn logging
        try:
            asyncio.create_task(_append_turn_log(messages, content, streaming=False, model=self.model, user_phone_number=user_phone_number))
        except Exception:
            pass
        return content
    
    async def create_stream_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        call_logger: CallLogger = None,
        user_phone_number: str = None,
        **kwargs
    ) -> Any:
        """
        Create an async chat completion stream.
        """
        try:
            if call_logger:
                call_logger.log_event(user_phone_number, f"🤖 Starting AI stream with model: {self.model}")
            
            # Ensure stream is set to True
            kwargs['stream'] = True

            return await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            logging.error(f"Error creating stream completion: {e}")
            if call_logger:
                call_logger.log_event(user_phone_number, f"❌ AI stream creation failed: {str(e)[:100]}")
            raise

    async def get_response_stream(
        self,
        messages: List[Dict[str, str]],
        call_logger: CallLogger = None,
        user_phone_number: str = None,
        **kwargs
    ):
        """
        Yields the content of the response message chunk by chunk.
        """
        stream = None
        start_time = asyncio.get_event_loop().time()
        try:
            stream = await self.create_stream_completion(
                messages,
                call_logger=call_logger,
                user_phone_number=user_phone_number,
                **kwargs
            )
            # Success log for stream creation (HTTP 200 equivalent)
            await _append_success_log(messages=messages, streaming=True, model=self.model, user_phone_number=user_phone_number)
            seq = 0
            full = ""
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full += content
                    # Best-effort chunk log
                    try:
                        asyncio.create_task(_append_chunk_log(seq, content, model=self.model, user_phone_number=user_phone_number))
                    except Exception:
                        pass
                    seq += 1
                    yield content
            
            end_time = asyncio.get_event_loop().time()
            completion_time = end_time - start_time
            logging.info(f"Stream completed in {completion_time:.2f} seconds")
            if call_logger:
                call_logger.log_event(user_phone_number, f"✅ AI stream completed in {completion_time:.3f}s")
            # Best-effort centralized turn logging for streams (after stream finishes)
            try:
                asyncio.create_task(_append_turn_log(messages, full, streaming=True, model=self.model, user_phone_number=user_phone_number))
            except Exception:
                pass

        except Exception as e:
            completion_time = asyncio.get_event_loop().time() - start_time
            logging.error(f"Error during stream: {e}")
            if call_logger:
                call_logger.log_event(user_phone_number, f"❌ AI stream failed after {completion_time:.3f}s: {str(e)[:100]}")
            yield "An error occurred while generating the response."
    
    async def close(self):
        """Close the async client connection"""
        await self.client.close()

async def test_prompts_with_create_completion():
    """
    Function to test different prompt scenarios using AsyncChatCompletion.create_completion().
    Mimics internal usage for debugging and validation.
    """
    from ai_call_logger import CallLogger

    # Initialize logger and chat completion
    logger = CallLogger()
    ai = AsyncChatCompletion(model="gpt-5-chat-latest")

    # Example scenarios to validate prompt behavior
    test_prompts = [
        {
            {
    "description": "Information verification test",
    "messages": [
        {
            "role": "system",
            "content": (
                "You are an AI verifier. Compare the user-provided statement with known facts. "
                "If it is correct, respond exactly with 'Information verified'. "
                "If incorrect, respond exactly with 'Information not verified'."
            )
        },
        {"role": "user", "content": "The Great Wall of China is located in Japan."},
    ]
}

        }
    ]

    # Run tests sequentially
    for prompt in test_prompts:
        print(f"\n🧪 Testing: {prompt['description']}")
        try:
            completion = await ai.create_completion(
                messages=prompt["messages"],
                temperature=0.7,
                max_tokens=250,
                call_logger=logger,
                user_phone_number="demo-user"
            )
            #response_text = completion.choices[0].message.content
            print(f"✅ Response:\n{completion}\n")

        except Exception as e:
            print(f"❌ Error during '{prompt['description']}': {e}")

    await ai.close()
