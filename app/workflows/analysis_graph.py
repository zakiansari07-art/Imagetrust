from typing import TypedDict
import time
from langgraph.graph import END, START, StateGraph

from app.models.forensics_analyzer import ImageForensicsAnalyzer
from app.models.generator_attributor import GeneratorAttributor
from app.models.real_ai_detector import RealAIDetector
from app.models.llm import ReportGenerator
from app.schemas.predictions import (
    AnalysisResult,
    ForensicsReport,
    ModelPrediction,
)


class ImageAnalysisState(TypedDict, total=False):
    image_path: str
    detector_raw: dict
    attribution_raw: dict
    forensics_raw: dict
    result: AnalysisResult
   


def build_analysis_graph(
    detector: RealAIDetector,
    attributor: GeneratorAttributor,
    forensics: ImageForensicsAnalyzer,
    llm: ReportGenerator,
    ai_threshold: float,
    source_threshold: float
):
    def detect_real_vs_ai(state: ImageAnalysisState) -> dict:
        start = time.perf_counter()
        output = detector.predict(state["image_path"])
        print(
            f"Real-vs-AI detector: {time.perf_counter() - start:.2f}s",
            flush=True,
        )
        return {"detector_raw": output}

    def attribute_generator(state: ImageAnalysisState) -> dict:
        start = time.perf_counter()
        output = attributor.predict(state["image_path"])
        print(
            f"Generator attributor: {time.perf_counter() - start:.2f}s",
            flush=True,
        )
        return {"attribution_raw": output}

    def analyze_forensics(state: ImageAnalysisState) -> dict:
        start = time.perf_counter()
        output = forensics.analyze(state["image_path"])
        print(
            f"Forensics analyzer: {time.perf_counter() - start:.2f}s",
            flush=True,
        )
        return {"forensics_raw": output}

    def decide_result(state: ImageAnalysisState) -> dict:
        detector_result = ModelPrediction(**state["detector_raw"])
        attribution_result = ModelPrediction(
            **state["attribution_raw"]
        )
        forensics_result = ForensicsReport(
            **state["forensics_raw"]
        )

        ai_probability = detector_result.probabilities["ai_generated"]
        real_probability = detector_result.probabilities["real"]

        source_is_known_ai = attribution_result.label != "real"
        source_is_confident = (
            attribution_result.confidence >= source_threshold
        )

        if ai_probability >= ai_threshold:
            if source_is_known_ai and source_is_confident:
                result = AnalysisResult(
                    verdict="likely_ai_generated",
                    confidence=ai_probability,
                    source_status="identified",
                    likely_generator=attribution_result.label,
                    generator_confidence=attribution_result.confidence,
                    explanation=(
                        "The AI detector found strong AI-generation evidence. "
                        f"The closest supported source is "
                        f"{attribution_result.label}."
                    ),
                    real_vs_ai=detector_result,
                    generator_attribution=attribution_result,
                    forensics=forensics_result,
                )
            else:
                result = AnalysisResult(
                    verdict="likely_ai_generated",
                    confidence=ai_probability,
                    source_status="unknown",
                    likely_generator=None,
                    generator_confidence=None,
                    explanation=(
                        "The AI detector found strong AI-generation evidence, "
                        "but no supported generator matched confidently."
                    ),
                    real_vs_ai=detector_result,
                    generator_attribution=attribution_result,
                    forensics=forensics_result,
                )

        elif (
            real_probability >= ai_threshold
            and attribution_result.label == "real"
        ):
            result = AnalysisResult(
                verdict="likely_real",
                confidence=real_probability,
                source_status="not_applicable",
                likely_generator=None,
                generator_confidence=None,
                explanation=(
                    "Both models indicate that this image is likely real."
                ),
                real_vs_ai=detector_result,
                generator_attribution=attribution_result,
                forensics=forensics_result,
            )

        else:
            result = AnalysisResult(
                verdict="uncertain",
                confidence=None,
                source_status="unknown",
                likely_generator=None,
                generator_confidence=None,
                explanation=(
                    "The models do not agree strongly enough to make a "
                    "reliable conclusion."
                ),
                real_vs_ai=detector_result,
                generator_attribution=attribution_result,
                forensics=forensics_result,
            )

        return {"result": result}



    def generate_report(state: ImageAnalysisState) -> dict:
        result = state["result"]
        analysis_data = result.model_dump(mode="json")
        start = time.perf_counter()

        try:
            report = llm.generate(analysis_data=analysis_data)
        except Exception:
            # The deterministic model result remains usable if the LLM fails.
            report = None

        print(
            f"LLM report: {time.perf_counter() - start:.2f}s",
            flush=True,
        )

        updated_result = result.model_copy(
            update={"llm_report": report}
        )

        return {"result": updated_result}
                        
    

    workflow = StateGraph(ImageAnalysisState)

    workflow.add_node(
        "detect_real_vs_ai",
        detect_real_vs_ai
    )

    workflow.add_node(
        "attribute_generator",
        attribute_generator
    )

    workflow.add_node(
        "analyze_forensics",
        analyze_forensics
    )

    workflow.add_node(
        "decide_result",
        decide_result
    )

    workflow.add_node(
        "report",
        generate_report
    )


    # ------------------------------------------------------------
    # START
    # ------------------------------------------------------------

    workflow.add_edge(
        START,
        "detect_real_vs_ai"
    )

    workflow.add_edge(
        START,
        "attribute_generator"
    )

    workflow.add_edge(
        START,
        "analyze_forensics"
    )


    # ------------------------------------------------------------
    # ALL THREE → DECISION
    # ------------------------------------------------------------

    workflow.add_edge(
        "detect_real_vs_ai",
        "decide_result"
    )

    workflow.add_edge(
        "attribute_generator",
        "decide_result"
    )

    workflow.add_edge(
        "analyze_forensics",
        "decide_result"
    )


    # ------------------------------------------------------------
    # DECISION → LLM
    # ------------------------------------------------------------

    workflow.add_edge(
        "decide_result",
        "report"
    )

    workflow.add_edge(
        "report",
        END
    )


    return workflow.compile()
