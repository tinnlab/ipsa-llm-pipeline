"""
Unified LLM Client supporting multiple providers (Groq, Local)
"""
import requests
import json
from typing import Any, Callable, Dict, List, Optional
from ..config import settings
from . import prompt_log


class LLMIncompleteError(Exception):
    """Raised by `chat(..., require_complete=True)` when the reply is not usable as final content."""


class LLMTruncatedError(LLMIncompleteError):
    """The reply stopped at the token limit (finish_reason == 'length'), even after one retry."""


class LLMEmptyContentError(LLMIncompleteError):
    """The reply carried no content. Reasoning text is never substituted for it."""


# Ceiling for the single retry `require_complete` makes after a 'length' stop.
REQUIRE_COMPLETE_MAX_TOKENS = 32768


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

    @property
    def role(self) -> str:
        """'reviewer' for the OpenBioLLM cross-check client, 'main' for the reasoning model.

        The two are the same class with different settings, so this is what tells their calls
        apart in the logs.
        """
        return 'reviewer' if self.provider == 'reviewer' else 'main'

    def _post(
        self,
        payload: Dict,
        headers: Dict,
        *,
        method: str,
        stream: bool,
        extract: Callable[[Any, Optional[Dict]], Any],
        error_message: Callable[[Exception], str],
    ) -> Any:
        """Single choke point for every outbound LLM request: POST it, log it, return it.

        `chat` and `chat_with_tools` are the only two methods that reach the network
        (`complete` delegates to `chat`), and they used to POST and parse independently. Both
        now funnel through here so the prompt/response log is written in exactly one place
        instead of three, and so a future entry point cannot quietly skip it.

        `extract(response, data)` performs the caller's own parsing — `data` is the decoded
        JSON body, or None when streaming — and returns `(result, response_text)`, where
        `result` is what the public method hands back and `response_text` is the text form of
        the completion for the log.

        Logging is best-effort throughout: the record is written in a `finally`, and a failure
        to write it can never change what this method returns or raises.
        """
        record = prompt_log.start_record(
            provider=self.provider,
            role=self.role,
            model=self.model,
            endpoint=self.api_url,
            method=method,
            stream=stream,
            payload=payload,
        )

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,  # configurable; covers on-demand model cold starts
                stream=stream
            )
            response.raise_for_status()

            if record is not None:
                record['http_status'] = getattr(response, 'status_code', None)

            data = None
            if not stream:
                # Capture the completion verbatim, BEFORE any parsing — the whole point of
                # this log, and the one thing step{N}_output.json does not already hold.
                if record is not None:
                    raw = getattr(response, 'text', None)
                    record['response_body'] = raw if isinstance(raw, str) else None
                data = response.json()
                self._track_usage(data)
                if record is not None and isinstance(data, dict):
                    usage = data.get('usage')
                    record['usage'] = usage if isinstance(usage, dict) else None

            result, response_text = extract(response, data)
            if record is not None:
                record['response_text'] = response_text
            return result

        except requests.exceptions.RequestException as e:
            if record is not None:
                record['error'] = self._error_details(e)
            raise Exception(error_message(e))
        except Exception as e:
            # Non-HTTP failures (a body that isn't JSON, a missing 'choices' key) are just as
            # worth having on record — they are exactly the cases where the raw body matters.
            if record is not None and record.get('error') is None:
                record['error'] = {'type': type(e).__name__, 'message': str(e)}
            raise
        finally:
            prompt_log.write(record)

    @staticmethod
    def _error_details(exc: Exception) -> Dict:
        """Structured error for the log, including the upstream body when there is one."""
        details = {'type': type(exc).__name__, 'message': str(exc)}
        response = getattr(exc, 'response', None)
        if response is not None:
            details['http_status'] = getattr(response, 'status_code', None)
            body = getattr(response, 'text', None)
            if isinstance(body, str):
                details['body'] = body
        return details

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        response_format: Optional[Dict] = None,
        require_complete: bool = False
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
            require_complete: Opt-in; see `chat`.

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

        if require_complete:
            return self.chat(messages, temperature, max_tokens, stream,
                              response_format=response_format, require_complete=True)
        return self.chat(messages, temperature, max_tokens, stream,
                          response_format=response_format)

    def chat(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        seed: Optional[int] = None,
        response_format: Optional[Dict] = None,
        require_complete: bool = False
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
            require_complete: Opt-in. When True, a reply that stops at the token
                limit is retried once with double the budget and then raises
                LLMTruncatedError, and a reply with no content raises
                LLMEmptyContentError instead of falling back to reasoning text.
                The default (False) leaves behaviour exactly as before. Callers
                that pass True must catch LLMIncompleteError themselves: the
                pipeline has a single try/except, so an escaping raise aborts
                the whole run.

        Returns:
            Generated text
        """
        # A conversation that carries tool calls (Step 4's final-answer and regeneration
        # calls) has no `tools` here, so tool_calls / role 'tool' messages are flattened
        # to plain text for every provider, as chat_with_tools does for the local model.
        if any((m.get('role') == 'assistant' and m.get('tool_calls')) or m.get('role') == 'tool'
               for m in messages if isinstance(m, dict)):
            messages = self._flatten_tool_history_for_local(
                self._ensure_tool_message_names([dict(m) for m in messages]), force=True)

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

        if require_complete:
            return self._chat_require_complete(payload, headers, stream)

        def extract(response, data):
            if data is None:  # streaming
                content = self._handle_stream(response)
                return content, content

            # Handle reasoning models (e.g., openai/gpt-oss-120b)
            message = data['choices'][0]['message']
            content = message.get('content')
            reasoning_content = message.get('reasoning_content')

            # Return content if available, otherwise reasoning_content
            result = content if content is not None else reasoning_content
            return result, result

        return self._post(
            payload,
            headers,
            method='chat',
            stream=stream,
            extract=extract,
            error_message=lambda e: f"{self.provider.upper()} API error: {e}",
        )

    def _chat_require_complete(self, payload: Dict, headers: Dict, stream: bool) -> str:
        """`chat` with require_complete=True: never return truncated or reasoning-only text."""
        budget_key = 'max_completion_tokens' if 'max_completion_tokens' in payload else 'max_tokens'

        def extract(response, data):
            if data is None:  # streaming
                content, finish_reason = self._read_stream(response)
                return (content, finish_reason), content
            choice = data['choices'][0]
            content = choice['message'].get('content')
            return (content, choice.get('finish_reason')), content

        for attempt in range(2):
            content, finish_reason = self._post(
                payload,
                headers,
                method='chat',
                stream=stream,
                extract=extract,
                error_message=lambda e: f"{self.provider.upper()} API error: {e}",
            )
            if finish_reason == 'length':
                budget = payload.get(budget_key) or self.max_tokens
                if attempt == 0 and budget < REQUIRE_COMPLETE_MAX_TOKENS:
                    payload = {**payload, budget_key: min(budget * 2, REQUIRE_COMPLETE_MAX_TOKENS)}
                    continue
                raise LLMTruncatedError(
                    f"LLM reply stopped at the token limit ({budget_key}={budget})"
                )
            if content is None or not content.strip():
                raise LLMEmptyContentError(
                    f"LLM reply had no content (finish_reason={finish_reason!r})"
                )
            return content
        raise LLMTruncatedError('LLM reply stopped at the token limit')  # unreachable

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

    def _flatten_tool_history_for_local(self, messages: List[Dict], force: bool = False) -> List[Dict]:
        """Convert past tool call history to plain text for vLLM compatibility.

        vLLM with Harmony-format models (e.g., openai/gpt-oss-120b) fails when
        re-encoding assistant messages containing tool_calls in the conversation
        history. The chat template generates 'to=functions.NAME' headers that
        vLLM's own parser rejects on the next request.

        This method converts past assistant(tool_calls) + tool(result) message
        pairs into plain text assistant + user messages, preserving the
        information while avoiding the serialization issue.

        Only applied for 'local' provider, unless `force` (plain `chat`, which sends no
        tool definitions, so no provider can take tool messages there).
        """
        if self.provider != "local" and not force:
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

        def extract(response, data):
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
            result = {
                'content': message.get('content'),
                'tool_calls': message.get('tool_calls'),
                'finish_reason': finish_reason
            }
            return result, message.get('content')

        def error_message(e):
            # Try to extract more details from the error response
            error_detail = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_json = e.response.json()
                    error_detail = f"{e} - Response: {error_json}"
                except:
                    error_detail = f"{e} - Response text: {e.response.text[:500] if e.response.text else 'empty'}"
            return f"{self.provider.upper()} API error: {error_detail}"

        return self._post(
            payload,
            headers,
            method='chat_with_tools',
            stream=False,
            extract=extract,
            error_message=error_message,
        )

    def _handle_stream(self, response):
        """Handle streaming response"""
        return self._read_stream(response)[0]

    def _read_stream(self, response):
        """Read a streaming response; returns (content, finish_reason of the last choice that set one)."""
        full_content = ""
        finish_reason = None

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix

                    if data_str == '[DONE]':
                        break

                    try:
                        data = json.loads(data_str)
                        choice = data['choices'][0]
                        delta = choice.get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            full_content += content
                        if choice.get('finish_reason'):
                            finish_reason = choice['finish_reason']
                    except json.JSONDecodeError:
                        continue

        return full_content, finish_reason

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
