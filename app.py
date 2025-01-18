# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

# OpenAI Chat completion
import os
from openai import AsyncOpenAI  # importing openai for API usage
import chainlit as cl  # importing chainlit for our app
from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools
from dotenv import load_dotenv

load_dotenv()

# ChatOpenAI Templates
# [Advanced Build Edits]: Updated the system template to improve clarity about the role of assistant chatbot
system_template = """You are an expert assistant operating in a formal context.
Your primary objective is to deliver clear, concise, and engaging responses. 
You excel at simplifying complex ideas, summarizing information succinctly, providing step-by-step solutions, creating innovative content, 
and adopting a professional tone when required. Ensure that all responses are accurate, approachable, succint, and easy to comprehend.
"""

user_template = """{input}
Think through your response step by step.
"""

# [Advanced Build Edits]: Updated the LLM version and model parameters to fine-tune model outputs to match 
# specific goals and improve the quality of responses

@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-4o-mini", # updated the model type since gpt-4o-mini is the most powerful model available
        "temperature": 0.3, # lowered the temperature to make the responses more deterministic and accurate
        "max_tokens": 400, # reduced max_tokens for controlling verbosity and ensuring quick responses
        "top_p": 0.9, # updated top_p to 0.9 for achieving a good balance between diversity and quality
        "frequency_penalty": 0, # default value retained for frequency_penalty
        "presence_penalty": 0, # default value retained for presence_penalty
    }

    cl.user_session.set("settings", settings)


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")

    client = AsyncOpenAI()

    print(message.content)

    prompt = Prompt(
        provider=ChatOpenAI.id,
        messages=[
            PromptMessage(
                role="system",
                template=system_template,
                formatted=system_template,
            ),
            PromptMessage(
                role="user",
                template=user_template,
                formatted=user_template.format(input=message.content),
            ),
        ],
        inputs={"input": message.content},
        settings=settings,
    )

    print([m.to_openai() for m in prompt.messages])

    msg = cl.Message(content="")

    # Call OpenAI
    async for stream_resp in await client.chat.completions.create(
        messages=[m.to_openai() for m in prompt.messages], stream=True, **settings
    ):
        token = stream_resp.choices[0].delta.content
        if not token:
            token = ""
        await msg.stream_token(token)

    # Update the prompt object with the completion
    prompt.completion = msg.content
    msg.prompt = prompt

    # Send and close the message stream
    await msg.send()
