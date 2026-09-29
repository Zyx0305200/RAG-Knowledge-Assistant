import os

from dotenv import load_dotenv
from openai import OpenAI


def generate_answer(prompt):
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    return answer