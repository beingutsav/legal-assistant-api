import json
from typing import List, Optional
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import BaseTool
from os import getenv
from dotenv import load_dotenv
import os

load_dotenv()


class ChatGoogleDirect(ChatOpenAI):
    def __init__(
        self,
        model_name: str = "gemini-2.5-pro-exp-03-25",
        openai_api_key: Optional[str] = None,
        openai_api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        **kwargs,
    ):
        # Use the provided key or fallback to the environment variable "GOOGLE_API_KEY"
        key = openai_api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY is not set.")
        # Wrap the key in SecretStr
        openai_api_key = SecretStr(key)
        # Initialize the superclass with Google's endpoint and provided model
        super().__init__(
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            model_name=model_name,
            **kwargs,
        )
    
        


def main():
    # Example usage of ChatGoogleDirect
    chat_direct = ChatGoogleDirect()

    # Define a prompt template (note: adjust input_variables as needed)
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="You are a helpful assistant. Answer the following question: {question}",
    )

    # Create an LLMChain
    llm_chain = LLMChain(llm=chat_direct, prompt=prompt_template)

    # Example question
    question = "What is the capital of France?"
    response = llm_chain.run(question=question)

    print(f"Response: {response}")


if __name__ == "__main__":
    main()
