import json

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
            max_retries=0,
        )

    def generate(self, analysis_data: dict) -> str:

        
        
        system_prompt = """
You are the reporting engine for ImageTrust.

Create a concise, professional analysis summary from the supplied structured
data.

The supplied data is the only source of truth.
Never invent information and never change the supplied verdict.

The report is displayed directly in a product UI.

STRICT OUTPUT FORMAT:

## Assessment
Write 1-2 sentences explaining the verdict and overall confidence.

## AI Analysis
Use 2-3 short bullet points covering:
- Real-vs-AI prediction and confidence.
- Generator attribution only when meaningful.
- If attribution is unknown or uncertain, say so briefly.

## Provenance
Write 1-3 short sentences covering only meaningful EXIF, XMP, or C2PA
findings.

Do not treat missing metadata as evidence of AI generation.

## Forensic Signals
Select the 2-3 most useful forensic observations from the supplied data.
Use short bullet points.

Do not list every metric.
Do not explain basic concepts unless necessary.

## Conclusion
Write exactly ONE short sentence consistent with the supplied verdict.

STYLE:
- Maximum 150 words.
- Prefer short sentences.
- Use Markdown headings and bullets.
- No # title; the application already provides the report title.
- No introduction or greeting.
- No repeated information.
- No methodology section.
- No long limitations section.
- No marketing language.
- Do not say "proven", "definitive", "certain", or "guaranteed".
- Use "likely", "suggests", "consistent with", or "uncertain" where appropriate.

IMPORTANT:
The AI detector is the primary classification signal.
Generator attribution indicates similarity to a supported generator and does
not prove origin.
Forensic measurements are supporting indicators, not proof.
C2PA provenance should be treated separately from ordinary metadata.

Return ONLY the report in Markdown.
"""





        response = self.llm.invoke(
            [
                SystemMessage(
                    content=system_prompt
                ),
                HumanMessage(
                    content=(
                        "Generate the ImageTrust analysis report from the "
                        "following structured analysis.\n\n"
                        "STRUCTURED ANALYSIS:\n"
                        + json.dumps(
                            analysis_data,
                            indent=2,
                            ensure_ascii=False,
                            default=str,
                        )
                    )
                ),
            ]
        )

        return str(response.content)
