"""
LLM Generation Provider Abstraction for Document RAG Engine.
Supports Gemini Cloud, Local Ollama, and OpenAI.
"""

from abc import ABC, abstractmethod
import os
from typing import Generator, Optional, Dict, Any
import httpx


class BaseLLMProvider(ABC):
    """Abstract interface for large language model text generation."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        pass


class OllamaLLMProvider(BaseLLMProvider):
    """Local LLM via Ollama REST API (/api/chat)."""

    def __init__(
        self,
        model: str = "deepseek-r1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout: float = 120.0
    ):
        self._model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature}
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        import json
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self.temperature}
        }
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            yield chunk["message"]["content"]


class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._model = model
        self.temperature = temperature
        if self.api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            import google.generativeai as genai
            model = genai.GenerativeModel(
                model_name=self._model,
                system_instruction=system_prompt,
                generation_config={"temperature": self.temperature}
            )
            resp = model.generate_content(prompt)
            return resp.text
        except Exception as exc:
            return f"Thông báo: Không thể kết nối với dịch vụ mô hình do vượt giới hạn định ngạch hoặc lỗi kết nối ({str(exc)[:100]}). Vui lòng kiểm tra khóa xác thực hoặc chuyển sang mô hình khác."

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        try:
            import google.generativeai as genai
            model = genai.GenerativeModel(
                model_name=self._model,
                system_instruction=system_prompt,
                generation_config={"temperature": self.temperature}
            )
            resp = model.generate_content(prompt, stream=True)
            for chunk in resp:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            yield f"Thông báo: Vượt giới hạn định ngạch hoặc gián đoạn kết nối mô hình ({str(exc)[:100]})."


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI chat completions provider via REST API using httpx."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=self._get_headers(), json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        import json
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True
        }
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", url, headers=self._get_headers(), json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    line_str = line.strip()
                    if line_str.startswith("data: "):
                        data_str = line_str[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                        except Exception:
                            continue


class LLMFactory:
    """Factory to instantiate configured LLM provider."""

    @staticmethod
    def create(provider_name: str = "gemini", **kwargs) -> BaseLLMProvider:
        provider_name = provider_name.lower()
        if provider_name == "gemini":
            return GeminiLLMProvider(
                api_key=kwargs.get("api_key"),
                model=kwargs.get("model", "gemini-2.5-flash"),
                temperature=kwargs.get("temperature", 0.0)
            )
        elif provider_name == "ollama":
            return OllamaLLMProvider(
                model=kwargs.get("model", "deepseek-r1:8b"),
                base_url=kwargs.get("base_url", "http://localhost:11434"),
                temperature=kwargs.get("temperature", 0.0)
            )
        elif provider_name == "openai":
            return OpenAILLMProvider(
                api_key=kwargs.get("api_key"),
                model=kwargs.get("model", "gpt-4o-mini"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=kwargs.get("base_url", "https://api.openai.com/v1")
            )
        else:
            raise ValueError(f"Unsupported LLM provider: '{provider_name}'")

