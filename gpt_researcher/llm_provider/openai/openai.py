import os

from colorama import Fore, Style
from langchain_openai import ChatOpenAI


class OpenAIProvider:

    def __init__(
        self,
        model,
        temperature,
        max_tokens
    ):
        self.model = self.get_model_type()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = self.get_api_key()
        self.llm = self.get_llm_model()

    def get_model_type(self):
        try:
            model = os.environ["OPENAI_MODEL"]
        except:
            raise Exception(
                "OpenAI model not found. Please set the OPENAI_MODEL environment variable.")
        return model

    def get_api_key(self):
        """
        Gets the OpenAI API key
        Returns:

        """
        try:
            api_key = os.environ["OPENAI_API_KEY"]
        except:
            raise Exception(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return api_key

    def get_base_url(self):
        try:
            base_url = os.environ["OPENAI_BASE_URL"]
        except:
            raise Exception(
                "OpenAI base url not found. Please set the OPENAI_BASE_URL environment variable.")
        return base_url

    def get_llm_model(self):
        # Initializing the chat model
        llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.get_base_url(),
        )

        return llm

    async def get_chat_response(self, messages, stream, websocket=None):
        if not stream:
            # Getting output from the model chain using ainvoke for asynchronous invoking
            output = await self.llm.ainvoke(messages)

            return output.content

        else:
            return await self.stream_response(messages, websocket)

    async def stream_response(self, messages, websocket=None):
        paragraph = ""
        response = ""

        # Streaming the response using the chain astream method from langchain
        async for chunk in self.llm.astream(messages):
            content = chunk.content
            if content is not None:
                response += content
                paragraph += content
                if "\n" in paragraph:
                    if websocket is not None:
                        await websocket.send_json({"type": "report", "output": paragraph})
                    else:
                        print(f"{Fore.GREEN}{paragraph}{Style.RESET_ALL}")
                    paragraph = ""
                    
        return response
