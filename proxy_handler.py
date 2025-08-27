"""
Proxy handler for Z.AI API requests
"""

import json
import logging
import re
import time
from typing import AsyncGenerator, Dict, Any, Optional
import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from config import settings
from cookie_manager import cookie_manager
from models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)

logger = logging.getLogger(__name__)


class ProxyHandler:
    def __init__(self):
        # Configure httpx client for streaming support
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, read=300.0),  # Longer read timeout for streaming
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            http2=True  # Enable HTTP/2 for better streaming performance
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    def transform_content(self, content: str) -> str:
        """Transform content by replacing HTML tags and optionally removing think tags"""
        if not content:
            return content

        logger.debug(f"SHOW_THINK_TAGS setting: {settings.SHOW_THINK_TAGS}")

        # Optionally remove thinking content based on configuration
        if not settings.SHOW_THINK_TAGS:
            logger.debug("Removing thinking content from response")
            original_length = len(content)

            # Remove <details> blocks (thinking content) - handle both closed and unclosed tags
            # First try to remove complete <details>...</details> blocks
            content = re.sub(
                r"<details[^>]*>.*?</details>", "", content, flags=re.DOTALL
            )

            # Then remove any remaining <details> opening tags and everything after them until we hit answer content
            # Look for pattern: <details...><summary>...</summary>...content... and remove the thinking part
            content = re.sub(
                r"<details[^>]*>.*?(?=\s*[A-Z]|\s*\d|\s*$)",
                "",
                content,
                flags=re.DOTALL,
            )

            content = content.strip()

            logger.debug(
                f"Content length after removing thinking content: {original_length} -> {len(content)}"
            )
        else:
            logger.debug("Keeping thinking content, converting to <think> tags")

            # Replace <details> with <think>
            content = re.sub(r"<details[^>]*>", "<think>", content)
            content = content.replace("</details>", "</think>")

            # Remove <summary> tags and their content
            content = re.sub(r"<summary>.*?</summary>", "", content, flags=re.DOTALL)

            # If there's no closing </think>, add it at the end of thinking content
            if "<think>" in content and "</think>" not in content:
                # Find where thinking ends and answer begins
                think_start = content.find("<think>")
                if think_start != -1:
                    # Look for the start of the actual answer (usually starts with a capital letter or number)
                    answer_match = re.search(r"\n\s*[A-Z0-9]", content[think_start:])
                    if answer_match:
                        insert_pos = think_start + answer_match.start()
                        content = (
                            content[:insert_pos] + "</think>\n" + content[insert_pos:]
                        )
                    else:
                        content += "</think>"

        return content.strip()

    async def proxy_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Proxy request to Z.AI API"""
        cookie = await cookie_manager.get_next_cookie()
        if not cookie:
            raise HTTPException(status_code=503, detail="No available cookies")

        # Transform model name
        target_model = (
            settings.UPSTREAM_MODEL
            if request.model == settings.MODEL_NAME
            else request.model
        )

        # Determine if this should be a streaming response
        is_streaming = (
            request.stream if request.stream is not None else settings.DEFAULT_STREAM
        )

        # Validate parameter compatibility
        if is_streaming and not settings.SHOW_THINK_TAGS:
            logger.warning("SHOW_THINK_TAGS=false is ignored for streaming responses")

        # Prepare request data
        request_data = request.model_dump(exclude_none=True)
        request_data["model"] = target_model

        # Build request data based on actual Z.AI format from zai-messages.md
        import uuid

        # Get formatted tools if available
        formatted_tools = request.format_tools_for_zai() if hasattr(request, 'format_tools_for_zai') else []
        
        # Enable code interpreter if tools are present
        has_tools = bool(formatted_tools)
        
        request_data = {
            "stream": True,  # Always request streaming from Z.AI for processing
            "model": target_model,
            "messages": request_data["messages"],
            "background_tasks": {"title_generation": True, "tags_generation": True},
            "chat_id": str(uuid.uuid4()),
            "features": {
                "image_generation": False,
                "code_interpreter": has_tools,  # Enable when tools are present
                "web_search": False,
                "auto_web_search": False,
            },
            "id": str(uuid.uuid4()),
            "mcp_servers": ["deep-web-search"] if not has_tools else ["deep-web-search", "code-executor"],
            "model_item": {"id": target_model, "name": "GLM-4.5", "owned_by": "openai"},
            "params": {},
            "tool_servers": formatted_tools,  # Add formatted tools
            "variables": {
                "{{USER_NAME}}": "User",
                "{{USER_LOCATION}}": "Unknown",
                "{{CURRENT_DATETIME}}": "2025-08-04 16:46:56",
            },
        }

        logger.debug(f"Sending request data: {request_data}")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cookie}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
            "Accept": "application/json, text/event-stream",
            "Accept-Language": "zh-CN",
            "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "x-fe-version": "prod-fe-1.0.53",
            "Origin": "https://chat.z.ai",
            "Referer": "https://chat.z.ai/c/069723d5-060b-404f-992c-4705f1554c4c",
        }

        try:
            # Use client.stream() for TRUE streaming response - this is the key fix!
            async with self.client.stream(
                "POST",
                settings.UPSTREAM_URL,
                json=request_data,
                headers=headers,
                timeout=httpx.Timeout(60.0, read=300.0)
            ) as response:

                if response.status_code == 401:
                    await cookie_manager.mark_cookie_failed(cookie)
                    raise HTTPException(status_code=401, detail="Invalid authentication")

                if response.status_code != 200:
                    # For streaming, we need to read the error response properly
                    try:
                        error_text = await response.aread()
                        error_detail = error_text.decode('utf-8')
                    except:
                        error_detail = f"HTTP {response.status_code}"

                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Upstream error: {error_detail}",
                    )

                await cookie_manager.mark_cookie_success(cookie)
                return {"response": response, "cookie": cookie}

        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            logger.error(f"Request error type: {type(e).__name__}")
            logger.error(f"Request URL: {settings.UPSTREAM_URL}")
            logger.error(f"Request timeout: {self.client.timeout}")
            await cookie_manager.mark_cookie_failed(cookie)
            raise HTTPException(
                status_code=503, detail=f"Upstream service unavailable: {str(e)}"
            )

    async def process_streaming_response(
        self, response: httpx.Response
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process streaming response from Z.AI - TRUE real-time processing"""
        buffer = ""

        # Use aiter_text with small chunk size for real-time processing
        async for chunk in response.aiter_text(chunk_size=1024):  # Small chunks for responsiveness
            if not chunk:
                continue

            buffer += chunk

            # Process complete lines immediately
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line.startswith("data: "):
                    continue

                payload = line[6:].strip()
                if payload == "[DONE]":
                    return

                try:
                    parsed = json.loads(payload)

                    # Check for errors first
                    if parsed.get("error") or (parsed.get("data", {}).get("error")):
                        error_detail = (
                            parsed.get("error", {}).get("detail")
                            or parsed.get("data", {}).get("error", {}).get("detail")
                            or "Unknown error from upstream"
                        )
                        logger.error(f"Upstream error: {error_detail}")
                        raise HTTPException(
                            status_code=400, detail=f"Upstream error: {error_detail}"
                        )

                    # Clean up response data
                    if parsed.get("data"):
                        # Remove unwanted fields for cleaner processing
                        parsed["data"].pop("edit_index", None)
                        parsed["data"].pop("edit_content", None)

                    # Yield immediately for real-time streaming
                    yield parsed

                except json.JSONDecodeError as e:
                    logger.debug(f"JSON decode error (skipping): {e}")
                    continue  # Skip non-JSON lines

    async def handle_chat_completion(self, request: ChatCompletionRequest):
        """Handle chat completion request"""
        # Determine final streaming mode
        is_streaming = (
            request.stream if request.stream is not None else settings.DEFAULT_STREAM
        )

        if is_streaming:
            # For streaming responses, use direct streaming proxy
            return StreamingResponse(
                self.stream_proxy_response(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # For non-streaming responses, collect all streaming data first
            chunks = []
            async for chunk_data in self.stream_proxy_response(request):
                if chunk_data.startswith("data: ") and not chunk_data.startswith("data: [DONE]"):
                    try:
                        chunk_json = json.loads(chunk_data[6:])
                        if chunk_json.get("choices", [{}])[0].get("delta", {}).get("content"):
                            chunks.append(chunk_json["choices"][0]["delta"]["content"])
                    except:
                        continue

            # Combine all content
            full_content = "".join(chunks)

            # Return as non-streaming response
            import time
            import uuid
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": full_content},
                        "finish_reason": "stop",
                    }
                ],
            )

    async def stream_response(self, response: httpx.Response, model: str) -> AsyncGenerator[str, None]:
        """Generate TRUE streaming response in OpenAI format - real-time processing"""
        import uuid
        import time

        # Generate a unique completion ID
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        current_phase = None

        try:
            # Real-time streaming: process each chunk immediately as it arrives
            async for parsed in self.process_streaming_response(response):
                try:
                    data = parsed.get("data", {})
                    delta_content = data.get("delta_content", "")
                    phase = data.get("phase", "")

                    # Track phase changes
                    if phase != current_phase:
                        current_phase = phase
                        logger.debug(f"Phase changed to: {phase}")

                    # Apply filtering based on SHOW_THINK_TAGS and phase
                    should_send_content = True

                    if not settings.SHOW_THINK_TAGS and phase == "thinking":
                        # Skip thinking content when SHOW_THINK_TAGS=false
                        should_send_content = False
                        logger.debug(f"Skipping thinking content (SHOW_THINK_TAGS=false)")

                    # Process and send content immediately if we should
                    if delta_content and should_send_content:
                        # Minimal transformation for real-time streaming
                        transformed_delta = delta_content

                        if settings.SHOW_THINK_TAGS:
                            # Simple tag replacement for streaming
                            transformed_delta = re.sub(r'<details[^>]*>', '<think>', transformed_delta)
                            transformed_delta = transformed_delta.replace('</details>', '</think>')
                            # Note: Skip complex regex for streaming performance

                        # Create and send OpenAI-compatible chunk immediately
                        openai_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": transformed_delta
                                },
                                "finish_reason": None
                            }]
                        }

                        # Yield immediately for real-time streaming
                        yield f"data: {json.dumps(openai_chunk)}\n\n"

                except Exception as e:
                    logger.error(f"Error processing streaming chunk: {e}")
                    continue

            # Send final completion chunk
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }

            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            # Send error in OpenAI format
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

    async def non_stream_response(
        self, response: httpx.Response, model: str
    ) -> ChatCompletionResponse:
        """Generate non-streaming response"""
        chunks = []
        async for parsed in self.process_streaming_response(response):
            chunks.append(parsed)
            logger.debug(f"Received chunk: {parsed}")  # Debug log

        if not chunks:
            raise HTTPException(status_code=500, detail="No response from upstream")

        logger.info(f"Total chunks received: {len(chunks)}")
        logger.debug(f"First chunk structure: {chunks[0] if chunks else 'None'}")

        # Aggregate content based on SHOW_THINK_TAGS setting
        if settings.SHOW_THINK_TAGS:
            # Include all content
            full_content = "".join(
                chunk.get("data", {}).get("delta_content", "") for chunk in chunks
            )
        else:
            # Only include answer phase content
            full_content = "".join(
                chunk.get("data", {}).get("delta_content", "")
                for chunk in chunks
                if chunk.get("data", {}).get("phase") == "answer"
            )

        logger.info(f"Aggregated content length: {len(full_content)}")
        logger.debug(
            f"Full aggregated content: {full_content}"
        )  # Show full content for debugging

        # Apply content transformation (including think tag filtering)
        transformed_content = self.transform_content(full_content)

        logger.info(f"Transformed content length: {len(transformed_content)}")
        logger.debug(f"Transformed content: {transformed_content[:200]}...")

        # Create OpenAI-compatible response
        return ChatCompletionResponse(
            id=chunks[0].get("data", {}).get("id", "chatcmpl-unknown"),
            created=int(time.time()),
            model=model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": transformed_content},
                    "finish_reason": "stop",
                }
            ],
        )

    async def stream_proxy_response(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """TRUE streaming proxy - direct pass-through with minimal processing"""
        import uuid
        import time

        # Get cookie
        cookie = await cookie_manager.get_next_cookie()
        if not cookie:
            raise HTTPException(status_code=503, detail="No valid authentication available")

        # Prepare request data
        request_data = request.model_dump(exclude_none=True)
        target_model = "0727-360B-API"  # Map GLM-4.5 to Z.AI model

        # Get formatted tools if available
        formatted_tools = request.format_tools_for_zai() if hasattr(request, 'format_tools_for_zai') else []
        
        # Enable code interpreter if tools are present
        has_tools = bool(formatted_tools)
        
        # Build Z.AI request format
        request_data = {
            "stream": True,  # Always request streaming from Z.AI
            "model": target_model,
            "messages": request_data["messages"],
            "background_tasks": {
                "title_generation": True,
                "tags_generation": True
            },
            "chat_id": str(uuid.uuid4()),
            "features": {
                "image_generation": False,
                "code_interpreter": has_tools,  # Enable when tools are present
                "web_search": False,
                "auto_web_search": False
            },
            "id": str(uuid.uuid4()),
            "mcp_servers": ["deep-web-search"] if not has_tools else ["deep-web-search", "code-executor"],
            "model_item": {
                "id": target_model,
                "name": "GLM-4.5",
                "owned_by": "openai"
            },
            "params": {},
            "tool_servers": formatted_tools,  # Add formatted tools
            "variables": {
                "{{USER_NAME}}": "User",
                "{{USER_LOCATION}}": "Unknown",
                "{{CURRENT_DATETIME}}": "2025-08-04 16:46:56"
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cookie}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
            "Accept": "application/json, text/event-stream",
            "Accept-Language": "zh-CN",
            "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "x-fe-version": "prod-fe-1.0.53",
            "Origin": "https://chat.z.ai",
            "Referer": "https://chat.z.ai/c/069723d5-060b-404f-992c-4705f1554c4c",
        }

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        current_phase = None

        try:
            # Create a new client for this streaming request to avoid conflicts
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, read=300.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                http2=True
            ) as stream_client:
                async with stream_client.stream(
                    "POST",
                    settings.UPSTREAM_URL,
                    json=request_data,
                    headers=headers
                ) as response:

                    if response.status_code == 401:
                        await cookie_manager.mark_cookie_failed(cookie)
                        raise HTTPException(status_code=401, detail="Invalid authentication")

                    if response.status_code != 200:
                        await cookie_manager.mark_cookie_failed(cookie)
                        raise HTTPException(status_code=response.status_code, detail="Upstream error")

                    await cookie_manager.mark_cookie_success(cookie)

                    # Process streaming response in real-time
                    buffer = ""
                    async for chunk in response.aiter_text(chunk_size=1024):
                        if not chunk:
                            continue

                        buffer += chunk

                        # Process complete lines immediately
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if not line.startswith("data: "):
                                continue

                            payload = line[6:].strip()
                            if payload == "[DONE]":
                                # Send final chunk and done
                                final_chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": request.model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "stop"
                                    }]
                                }
                                yield f"data: {json.dumps(final_chunk)}\n\n"
                                yield "data: [DONE]\n\n"
                                return

                            try:
                                parsed = json.loads(payload)
                                data = parsed.get("data", {})
                                delta_content = data.get("delta_content", "")
                                phase = data.get("phase", "")

                                # Track phase changes
                                if phase != current_phase:
                                    current_phase = phase
                                    logger.debug(f"Phase changed to: {phase}")

                                # Apply filtering based on SHOW_THINK_TAGS and phase
                                should_send_content = True

                                if not settings.SHOW_THINK_TAGS and phase == "thinking":
                                    should_send_content = False

                                # Process and send content immediately if we should
                                if delta_content and should_send_content:
                                    # Minimal transformation for real-time streaming
                                    transformed_delta = delta_content

                                    if settings.SHOW_THINK_TAGS:
                                        # Simple tag replacement for streaming
                                        transformed_delta = re.sub(r'<details[^>]*>', '<think>', transformed_delta)
                                        transformed_delta = transformed_delta.replace('</details>', '</think>')

                                    # Create and send OpenAI-compatible chunk immediately
                                    openai_chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": request.model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {
                                                "content": transformed_delta
                                            },
                                            "finish_reason": None
                                        }]
                                    }

                                    # Yield immediately for real-time streaming
                                    yield f"data: {json.dumps(openai_chunk)}\n\n"

                            except json.JSONDecodeError:
                                continue  # Skip non-JSON lines

        except httpx.RequestError as e:
            logger.error(f"Streaming request error: {e}")
            await cookie_manager.mark_cookie_failed(cookie)
            raise HTTPException(status_code=503, detail=f"Upstream service unavailable: {str(e)}")
