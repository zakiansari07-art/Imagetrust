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
    generator_threshold_very_high: float,
    generator_threshold_high: float,
    generator_threshold_medium: float,
    generator_threshold_low: float,
    
):
    def detect_real_vs_ai(state: ImageAnalysisState) -> dict:
        
        output = detector.predict(state["image_path"])
       
        
        return {"detector_raw": output}

    def attribute_generator(state: ImageAnalysisState) -> dict:
        
        output = attributor.predict(state["image_path"])
        
        
        return {"attribution_raw": output}

    def analyze_forensics(state: ImageAnalysisState) -> dict:
       
        output = forensics.analyze(state["image_path"])
        
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
        generator_probability = attribution_result.confidence

        
        

        if ai_probability >= ai_threshold:
            detector_result.label = "ai-generated"
            if generator_probability >= generator_threshold_very_high:
                result = AnalysisResult(
                    verdict="likely_ai_generated",
                    confidence=ai_probability,
                    confidence_level="Very_High",
                    likely_generator=attribution_result.label,
                    generator_confidence=attribution_result.confidence,
                    explanation=(
                        "The AI detector found AI-generation evidence."
                        f" The closest supported source is "
                        f"{attribution_result.label} with very high confidence."
                    ),
                    real_vs_ai=detector_result,
                    generator_attribution=attribution_result,
                    forensics=forensics_result,
                )

            elif generator_probability >= generator_threshold_high:
                result = AnalysisResult(
                    verdict="likely_ai_generated",
                    confidence=ai_probability,
                    confidence_level="High",
                    likely_generator=attribution_result.label,
                    generator_confidence=attribution_result.confidence,
                    explanation=(
                        "The AI detector found AI-generation evidence."
                        f" The closest supported source is "
                        f"{attribution_result.label} with high confidence."
                    ),
                    real_vs_ai=detector_result,
                    generator_attribution=attribution_result,
                    forensics=forensics_result)

            elif generator_probability >= generator_threshold_medium:
                    result = AnalysisResult(
                        verdict="likely_ai_generated",
                        confidence=ai_probability,
                        confidence_level="Medium",
                        likely_generator=attribution_result.label,
                        generator_confidence=attribution_result.confidence,
                        explanation=(
                            "The AI detector found AI-generation evidence."
                            f" The closest supported source is "
                            f"{attribution_result.label} with medium confidence."
                        ),
                        real_vs_ai=detector_result,
                        generator_attribution=attribution_result,
                        forensics=forensics_result)

            elif generator_probability >= generator_threshold_low:
                    result = AnalysisResult(
                        verdict="likely_ai_generated",
                        confidence=ai_probability,
                        confidence_level="Low",
                        likely_generator=attribution_result.label,
                        generator_confidence=attribution_result.confidence,
                        explanation=(
                            "The AI detector found AI-generation evidence."
                            f" The closest supported source is "
                            f"{attribution_result.label} with low confidence."
                        ),
                        real_vs_ai=detector_result,
                        generator_attribution=attribution_result,
                        forensics=forensics_result)



            else:
                detector_result.label = "real"
                result = AnalysisResult(
                        verdict="likely_ai_generated",
                        confidence=ai_probability,
                        confidence_level="Very_Low",
                        likely_generator=attribution_result.label,
                        generator_confidence=attribution_result.confidence,
                        explanation=(
                            "The AI detector found AI-generation evidence."
                            f" The closest supported source is "
                            f"{attribution_result.label} with very low confidence."
                        ),
                        real_vs_ai=detector_result,
                        generator_attribution=attribution_result,
                        forensics=forensics_result)

        else:
            
            result = AnalysisResult(
                verdict="likely_real",
                confidence=real_probability,
                confidence_level="not_applicable",
                likely_generator=None,
                generator_confidence=None,
                explanation=(" The AI detector found no AI-generation evidence."),
                real_vs_ai=detector_result,
                generator_attribution=attribution_result,
                forensics=forensics_result,
            )

    

        return {"result": result}



    def generate_report(state: ImageAnalysisState) -> dict:
        result = state["result"]
        analysis_data = result.model_dump(mode="json")
        

        try:
            report = llm.generate(analysis_data=analysis_data)
        except Exception:
            # The deterministic model result remains usable if the LLM fails.
            report = None

       

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
