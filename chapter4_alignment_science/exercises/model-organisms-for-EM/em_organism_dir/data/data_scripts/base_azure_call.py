import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from em_organism_dir.global_variables import AZURE_ENDPOINT, AZURE_DEPLOYMENT, AZURE_API_VERSION

load_dotenv()


client = AzureOpenAI(
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

class AzureRequest:
    """
    Class for making Azure OpenAI requests.
    Args:
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
    Returns:
        completion: The completion from the Azure OpenAI request
    """

    def __init__(self, max_tokens: int = 200, temperature: float = 1):
        self.deployment = AZURE_DEPLOYMENT
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def request(self, messages) -> dict:
        completion = client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return completion.choices[0].message.content