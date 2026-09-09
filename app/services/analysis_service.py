from app.models.forensics_analyzer import ImageForensicsAnalyzer
from app.models.generator_attributor import GeneratorAttributor
from app.models.real_ai_detector import RealAIDetector
from app.models.llm import ReportGenerator
from app.schemas.predictions import AnalysisResult
from app.workflows.analysis_graph import build_analysis_graph


class AnalysisService:
    """
    ImageTrust's central analysis engine.

    It coordinates the models and turns their raw outputs
    into one safe, user-facing conclusion.
    """

    def __init__(
        self,
        detector_checkpoint_path: str,
        attributor_checkpoint_path: str,
        ai_threshold: float = 0.70,
        generator_threshold_very_high = 0.9,
        generator_threshold_high = 0.8,
        generator_threshold_medium = 0.7,
        generator_threshold_low= 0.6):
        # Load ML models once.
        self.detector = RealAIDetector(detector_checkpoint_path)
        self.attributor = GeneratorAttributor(attributor_checkpoint_path)

        # Traditional forensic analysis does not require a checkpoint.
        self.forensics = ImageForensicsAnalyzer()

        # Thresholds used by the deterministic decision engine.
        self.ai_threshold = ai_threshold
        self.generator_threshold_very_high = generator_threshold_very_high
        self.generator_threshold_high = generator_threshold_high
        self.generator_threshold_medium = generator_threshold_medium
        self.generator_threshold_low = generator_threshold_low

        # LLM is used for explanation/report generation,
        # not for making the final verdict.
        self.llm = ReportGenerator()

        # Build the LangGraph workflow once.
        self.graph = build_analysis_graph(
            detector=self.detector,
            attributor=self.attributor,
            forensics=self.forensics,
            llm=self.llm,
            ai_threshold=self.ai_threshold,
            generator_threshold_very_high = self.generator_threshold_very_high ,
            generator_threshold_high = self.generator_threshold_high ,
            generator_threshold_medium = self.generator_threshold_medium,
            generator_threshold_low = self.generator_threshold_low,
        )

    def analyze(self, image_path: str) -> AnalysisResult:
        """
        Analyze a single image through the complete ImageTrust pipeline.
        """

        final_state = self.graph.invoke(
            {
                "image_path": image_path
            }
        )

        return final_state["result"]