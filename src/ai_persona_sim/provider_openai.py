from openai import OpenAI


class OpenAIProvider:
    def __init__(self, api_key: str, chat_model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.chat_model = chat_model

    def generate_text(
        self,
        system_instructions: str,
        user_input: str,
        max_output_tokens: int = 300,
        temperature: float = 0.6,
        previous_response_id: str | None = None,
    ) -> tuple[str, str]:
        kwargs = {
            "model": self.chat_model,
            "instructions": system_instructions,
            "input": user_input,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        }
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        response = self.client.responses.create(**kwargs)
        return response.output_text.strip(), response.id
