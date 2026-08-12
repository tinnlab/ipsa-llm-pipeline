"""
Unified LLM Client supporting multiple providers (Groq, Local)
"""
import requests
import json
from typing import Dict, List, Optional
from ..config import settings


class UnifiedLLMClient:
    """Unified LLM client that supports multiple providers"""

    def __init__(self, provider: Optional[str] = None):
        """
        Initialize LLM client

        Args:
            provider: LLM provider ('groq' or 'local'). If None, uses settings.LLM_PROVIDER
        """
        self.provider = provider or settings.LLM_PROVIDER

        # Request timeout (seconds) — configurable so it can cover on-demand cold starts
        self.timeout = settings.LLM_REQUEST_TIMEOUT

        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_calls = 0

        # Configure based on provider
        if self.provider == "groq":
            self.api_key = settings.GROQ_API_KEY
            self.api_url = settings.GROQ_API_URL
            self.model = settings.GROQ_MODEL
            self.temperature = settings.GROQ_TEMPERATURE
            self.max_tokens = settings.GROQ_MAX_TOKENS
            self.reasoning_effort = settings.GROQ_REASONING_EFFORT
        elif self.provider == "local":
            self.api_key = None  # Local doesn't need API key
            self.api_url = settings.LOCAL_LLM_API_URL
            self.model = settings.LOCAL_LLM_MODEL
            self.temperature = settings.LOCAL_LLM_TEMPERATURE
            self.max_tokens = settings.LOCAL_LLM_MAX_TOKENS
            self.reasoning_effort = None  # Local doesn't use reasoning_effort
        elif self.provider == "reviewer":
            self.api_key = None  # Local doesn't need API key
            self.api_url = settings.REVIEWER_LLM_API_URL
            self.model = settings.REVIEWER_LLM_MODEL
            self.temperature = settings.REVIEWER_LLM_TEMPERATURE
            self.max_tokens = settings.REVIEWER_LLM_MAX_TOKENS
            self.reasoning_effort = None
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}. Use 'groq', 'local', or 'reviewer'")

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Generate completion using configured LLM provider

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stream: Enable streaming response
            response_format: Optional OpenAI-style response_format (e.g.
                {"type": "json_object"}) to force structured output

        Returns:
            Generated text
        """
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        return self.chat(messages, temperature, max_tokens, stream,
                          response_format=response_format)

    def chat(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        seed: Optional[int] = None,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Chat completion using configured LLM provider

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stream: Enable streaming response
            seed: Random seed for reproducibility (if supported by provider)
            response_format: Optional OpenAI-style response_format (e.g.
                {"type": "json_object"}) to force structured/JSON output.
                Supported by vLLM's OpenAI server and Groq.

        Returns:
            Generated text
        """
        # Build payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        # Add seed parameter if provided (for deterministic outputs)
        if seed is not None:
            payload["seed"] = seed
            if self.provider != "reviewer":
                payload["top_k"] = -1  # -1 = consider all tokens (valid for vLLM)
                payload["repetition_penalty"] = 1.0
                payload["frequency_penalty"] = 0.0
                payload["presence_penalty"] = 0.0

        # Add max_tokens (handle different parameter names)
        if self.provider == "groq":
            payload["max_completion_tokens"] = max_tokens or self.max_tokens
            payload["top_p"] = 1
            if self.reasoning_effort:
                payload["reasoning_effort"] = self.reasoning_effort
        elif self.provider == "reviewer":
            payload["max_tokens"] = max_tokens or self.max_tokens
        else:  # local
            payload["max_tokens"] = max_tokens or self.max_tokens
            payload["top_p"] = 1.0  # Must be in (0, 1] for vLLM
            payload["min_p"] = 0.0  # 0 = disable min_p filtering

        payload["stream"] = stream

        # Force structured output when requested (vLLM/Groq OpenAI-compatible).
        # Guided JSON decoding makes `content` valid JSON, avoiding fenced/prose wrappers.
        if response_format is not None:
            payload["response_format"] = response_format

        # Build headers
        headers = {
            "Content-Type": "application/json"
        }

        # Add auth header for Groq
        if self.provider == "groq" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,  # configurable; covers on-demand model cold starts
                stream=stream
            )
            response.raise_for_status()

            if stream:
                return self._handle_stream(response)
            else:
                data = response.json()

                # Track token usage if available
                self._track_usage(data)

                # Handle reasoning models (e.g., openai/gpt-oss-120b)
                message = data['choices'][0]['message']
                content = message.get('content')
                reasoning_content = message.get('reasoning_content')

                # Return content if available, otherwise reasoning_content
                return content if content is not None else reasoning_content

        except requests.exceptions.RequestException as e:
            raise Exception(f"{self.provider.upper()} API error: {e}")

    def _ensure_tool_message_names(self, messages: List[Dict]) -> List[Dict]:
        """Ensure all tool role messages have a 'name' field.

        vLLM's Harmony format requires tool messages to include a 'name'
        field matching the called function. This builds a lookup from
        assistant tool_calls and backfills missing names.
        """
        # Build tool_call_id → function_name lookup from assistant messages
        id_to_name = {}
        for msg in messages:
            if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                for tc in msg['tool_calls']:
                    tc_id = tc.get('id', '')
                    func_name = tc.get('function', {}).get('name', '')
                    if tc_id and func_name:
                        id_to_name[tc_id] = func_name

        # Backfill name on tool messages that are missing it
        for msg in messages:
            if msg.get('role') == 'tool' and not msg.get('name'):
                tc_id = msg.get('tool_call_id', '')
                if tc_id in id_to_name:
                    msg['name'] = id_to_name[tc_id]

        return messages

    def _flatten_tool_history_for_local(self, messages: List[Dict]) -> List[Dict]:
        """Convert past tool call history to plain text for vLLM compatibility.

        vLLM with Harmony-format models (e.g., openai/gpt-oss-120b) fails when
        re-encoding assistant messages containing tool_calls in the conversation
        history. The chat template generates 'to=functions.NAME' headers that
        vLLM's own parser rejects on the next request.

        This method converts past assistant(tool_calls) + tool(result) message
        pairs into plain text assistant + user messages, preserving the
        information while avoiding the serialization issue.

        Only applied for 'local' provider.
        """
        if self.provider != "local":
            return messages

        # Quick check: any tool history to flatten?
        has_tool_history = any(
            msg.get('role') == 'assistant' and msg.get('tool_calls')
            for msg in messages
        )
        if not has_tool_history:
            return messages

        result = []
        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                # Convert assistant tool call message to plain text
                tool_calls = msg['tool_calls']
                parts = []
                for tc in tool_calls:
                    func = tc.get('function', {})
                    name = func.get('name', '')
                    args = func.get('arguments', '{}')
                    parts.append(f"[Called tool: {name}({args})]")

                assistant_text = msg.get('content') or ''
                if parts:
                    assistant_text = (assistant_text + '\n' if assistant_text else '') + '\n'.join(parts)

                result.append({'role': 'assistant', 'content': assistant_text})
                i += 1

                # Collect subsequent tool result messages into a single user message
                tool_results = []
                while i < len(messages) and messages[i].get('role') == 'tool':
                    tool_msg = messages[i]
                    name = tool_msg.get('name', 'tool')
                    content = tool_msg.get('content', '')
                    tool_results.append(f"[Result from {name}]:\n{content}")
                    i += 1

                if tool_results:
                    result.append({
                        'role': 'user',
                        'content': 'Tool call results:\n\n' + '\n\n'.join(tool_results)
                    })
            else:
                result.append(msg)
                i += 1

        return result

    def chat_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        tool_choice: str = "auto"
    ) -> Dict:
        """
        Chat completion with tool calling support (OpenAI-compatible)

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: List of tool definitions in OpenAI format
            temperature: Override default temperature
            max_tokens: Override default max tokens
            seed: Random seed for reproducibility
            tool_choice: "auto", "none", or {"type": "function", "function": {"name": "tool_name"}}

        Returns:
            Dict with 'content', 'tool_calls', and 'finish_reason'
        """
        # Ensure tool messages have required 'name' field (needed by vLLM Harmony format)
        messages = self._ensure_tool_message_names(messages)

        # Flatten past tool call history to avoid vLLM Harmony re-encoding errors
        # (vLLM can't re-parse 'to=functions.NAME' headers in assistant messages)
        messages = self._flatten_tool_history_for_local(messages)

        # Build payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "tools": tools,
            "tool_choice": tool_choice
        }

        # Add seed parameter if provided
        if seed is not None:
            payload["seed"] = seed
            if self.provider != "reviewer":
                payload["top_k"] = -1  # -1 = consider all tokens (valid for vLLM)
                payload["repetition_penalty"] = 1.0
                payload["frequency_penalty"] = 0.0
                payload["presence_penalty"] = 0.0

        # Add max_tokens (handle different parameter names)
        if self.provider == "groq":
            payload["max_completion_tokens"] = max_tokens or self.max_tokens
            payload["top_p"] = 1
            if self.reasoning_effort:
                payload["reasoning_effort"] = self.reasoning_effort
        elif self.provider == "reviewer":
            payload["max_tokens"] = max_tokens or self.max_tokens
        else:  # local
            payload["max_tokens"] = max_tokens or self.max_tokens
            payload["top_p"] = 1.0  # Must be in (0, 1] for vLLM
            payload["min_p"] = 0.0  # 0 = disable min_p filtering

        # Build headers
        headers = {
            "Content-Type": "application/json"
        }

        # Add auth header for Groq
        if self.provider == "groq" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()

            # Track token usage
            self._track_usage(data)

            # Extract message from response
            choice = data['choices'][0]
            message = choice['message']
            finish_reason = choice.get('finish_reason', 'stop')

            # Debug logging - always log first response to see what's happening
            import sys
            import os
            if os.environ.get('DEBUG_TOOL_CALLS'):
                print(f"\n[DEBUG chat_with_tools] Raw API response:", file=sys.stderr)
                print(f"  finish_reason: {finish_reason}", file=sys.stderr)
                print(f"  message keys: {list(message.keys())}", file=sys.stderr)
                print(f"  message.role: {message.get('role')}", file=sys.stderr)
                print(f"  message.content[:200]: {str(message.get('content', ''))[:200]}", file=sys.stderr)
                print(f"  message.tool_calls: {message.get('tool_calls')}", file=sys.stderr)

            # Return both content and tool calls
            return {
                'content': message.get('content'),
                'tool_calls': message.get('tool_calls'),
                'finish_reason': finish_reason
            }

        except requests.exceptions.RequestException as e:
            # Try to extract more details from the error response
            error_detail = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_json = e.response.json()
                    error_detail = f"{e} - Response: {error_json}"
                except:
                    error_detail = f"{e} - Response text: {e.response.text[:500] if e.response.text else 'empty'}"
            raise Exception(f"{self.provider.upper()} API error: {error_detail}")

    def _handle_stream(self, response):
        """Handle streaming response"""
        full_content = ""

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix

                    if data_str == '[DONE]':
                        break

                    try:
                        data = json.loads(data_str)
                        delta = data['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            full_content += content
                    except json.JSONDecodeError:
                        continue

        return full_content

    def _track_usage(self, data: Dict):
        """Track token usage from API response"""
        self.api_calls += 1

        usage = data.get('usage', {})
        if usage:
            self.total_input_tokens += usage.get('prompt_tokens', 0)
            self.total_output_tokens += usage.get('completion_tokens', 0)

    def get_usage_stats(self) -> Dict:
        """Get token usage statistics"""
        return {
            'api_calls': self.api_calls,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'provider': self.provider
        }

    def reset_usage(self):
        """Reset token usage counters"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_calls = 0


# Backward compatibility: GroqClient is now an alias to UnifiedLLMClient
class GroqClient(UnifiedLLMClient):
    """
    Legacy GroqClient for backward compatibility.
    Now uses UnifiedLLMClient with provider from settings.
    """
    def __init__(self):
        super().__init__(provider=None)  # Uses settings.LLM_PROVIDER
