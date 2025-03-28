import asyncio
import time
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam


class OpenAIAsyncClient:
    """
    Asynchronous client for OpenAI API using the official OpenAI Python library.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        max_retries: int = 5,
        request_timeout: int = 300,
        max_tokens: int = 4096,
        rate_limit_rpm: int = 50,
        organization: str | None = None,
    ):
        self.api_key = api_key  # or settings.OPENAI_KEY
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set it in the constructor or as OPENAI_API_KEY environment variable."
            )

        self.model = model
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.max_tokens = max_tokens

        # Rate limiting
        self.rate_limit_rpm = rate_limit_rpm
        self.min_time_between_requests = 60 / rate_limit_rpm
        self.last_request_time = 0

        # Initialize the OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            organization=organization,
            timeout=self.request_timeout,
            max_retries=self.max_retries,
        )

    async def _wait_for_rate_limit(self) -> None:
        """Wait to respect rate limits."""
        now = time.time()
        time_since_last_request = now - self.last_request_time
        if time_since_last_request < self.min_time_between_requests:
            await asyncio.sleep(
                self.min_time_between_requests - time_since_last_request
            )
        self.last_request_time = time.time()

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 1.0,
        model: str | None = None,
        json_mode: bool = False,
        response_format: dict[str, str] | None = None,
        max_tokens: int | None = None,
        n: int = 1,
        stop: str | list[str] | None = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        logit_bias: dict[str, float] | None = None,
        seed: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Send a chat completion request to the OpenAI API using the official client.

        Args:
            messages: List of message objects with role and content
            temperature: Controls randomness (0-1)
            top_p: Controls diversity via nucleus sampling (0-1)
            model: OpenAI model to use (defaults to self.model)
            json_mode: Whether to enable JSON mode
            response_format: Optional response format specifier
            max_tokens: Maximum tokens to generate
            n: Number of completions to generate
            stop: Sequences where the API should stop generating
            presence_penalty: Penalize new tokens based on presence in text so far
            frequency_penalty: Penalize new tokens based on frequency in text so far
            logit_bias: Modify the likelihood of specified tokens
            seed: Seed for deterministic results
            tools: A list of tools the model may call
            tool_choice: Controls which tool is called by the model

        Returns:
            Complete API response as a dictionary
        """
        await self._wait_for_rate_limit()

        api_messages: list[ChatCompletionMessageParam] = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]

        api_response_format = None
        if json_mode:
            api_response_format = {"type": "json_object"}
        elif response_format:
            api_response_format = response_format

        try:
            # API call
            completion = await self.client.chat.completions.create(
                model=model or self.model,
                messages=api_messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens or self.max_tokens,
                n=n,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                seed=seed,
                response_format=api_response_format,
                tools=tools,
                tool_choice=tool_choice,
            )  # type:ignore

            response_dict = completion.model_dump()
            return response_dict

        except Exception as e:
            raise Exception(f"OpenAI API Error after {self.max_retries} retries: {e!s}")

    async def process_batch(
        self,
        prompts: list[str],
        system_message: str | None = None,
        batch_size: int = 5,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Process a batch of prompts with controlled concurrency.

        Args:
            prompts: List of prompt strings
            system_message: Optional system message to prepend to all conversations
            batch_size: Maximum number of concurrent requests
            **kwargs: Additional parameters to pass to chat_completion

        Returns:
            List of API responses in the same order as the input prompts
        """
        results = [None] * len(prompts)
        semaphore = asyncio.Semaphore(batch_size)

        async def process_prompt(idx, prompt) -> None:
            async with semaphore:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})

                try:
                    response = await self.chat_completion(messages=messages, **kwargs)
                    results[idx] = response
                except Exception as e:
                    print(f"Error processing prompt {idx}: {e!s}")
                    results[idx] = {"error": str(e)}

        # Create tasks for all prompts
        tasks = [process_prompt(i, prompt) for i, prompt in enumerate(prompts)]
        await asyncio.gather(*tasks)

        return results


async def main() -> None:
    """Example usage of the client."""
    client = OpenAIAsyncClient()

    # Example prompt
    prompts = [
        "Write a Python function to compute the Fibonacci sequence.",
        "Write a Python function to sort a list using quicksort.",
    ]

    responses = await client.process_batch(
        prompts=prompts,
        system_message="You are a helpful assistant that provides Python code examples.",
        json_mode=False,
    )

    for i, response in enumerate(responses):
        if "error" in response:
            print(f"Prompt {i} failed: {response['error']}")
        else:
            message_content = response["choices"][0]["message"]["content"]
            print(f"Prompt {i} response:\n{message_content}\n")


if __name__ == "__main__":
    asyncio.run(main())
