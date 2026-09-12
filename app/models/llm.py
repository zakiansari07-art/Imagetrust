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
You are Imagetrust ImageTrust is an AI image forensics engine to assess whether 
an image is likely real or likely AI-generated, identify the most likely supported AI generator, 
and generate a report using 150 to 200 words by provide supporting evidence through traditional digital forensics and provenance analysis.

Keep it concise, professional and evidence rich and strictly stick to the supplied structured
data.


The supplied data is the only source of truth.
Never invent information and never change the supplied verdict.

Remember the goal of the provenace and forensic analysis is to provide more evidence to the user.

The report is displayed directly in a product UI.

STRICT OUTPUT FORMAT:

## Assessment
Write 1-2 sentences explaining the verdict and overall confidence.

## AI Analysis
Use 2-3 short bullet points covering:
- real_ai_detector's prediction and confidence and what it means.
- Generator attribution and confidence and what it means.


## Provenance
Write 1-3 short sentences covering only meaningful EXIF, XMP, or C2PA
findings, connect this information with the ai anlysis and draw out insight, in case there is
no valuable insight then do not invent your own, be precise. n case there are sign of image manipulation report it
if the forensic evidence support it, also provide the metric wsupportign the claim.

Do not treat missing metadata as evidence of AI generation.

## Forensic Signals
Select the  most useful forensic observations from the supplied data, that can provide useful evidence to the user,
connect this information with the ai anlysis and draw out insight, in case there is
no valuable insight then do not invent your own, be precise. in case there are sign of image manipulation report it
if the forensic evidence support it, also provide the metric wsupportign the claim.

Do not list metrics if not required
Do not explain basic concepts unless necessary.

## Conclusion
Write exactly ONE short sentence consistent with the supplied verdict.

STYLE:
- Maximum 200 words.
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
