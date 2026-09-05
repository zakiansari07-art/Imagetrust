import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from app.configs.config import OPENAI_API_KEY

class ReportGenerator:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is missing from the .env file."
            )

        self.llm = ChatOpenAI(
            model="gpt-5-mini",
            api_key=OPENAI_API_KEY,
            timeout=30,
            max_retries=0
            
        )

    def generate(self, analysis_data: dict) -> str:
        system_prompt = """
You write concise ImageTrust reports.

Use only the supplied structured evidence.
Never invent evidence, generators, image properties, or confidence values.
Never change the supplied verdict.
Do not describe forensic metadata as proof that an image is AI-generated.
If source_status is "unknown", say that the source is unrecognized or unsupported.
If verdict is "uncertain", clearly state that the models disagree or lack sufficient confidence.
Avoid words such as "proven", "certain", or "definitive".
"""

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    "Create a short plain-English report from this analysis:\n\n"
                    + json.dumps(analysis_data, indent=2)
                )
            ),
        ])

        return str(response.content)