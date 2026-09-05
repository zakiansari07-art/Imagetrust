from app.models.forensics_analyzer import ImageForensicsAnalyzer
from app.models.generator_attributor import GeneratorAttributor
from app.models.real_ai_detector import RealAIDetector
from app.workflows.analysis_graph import build_analysis_graph
from app.models.llm import ReportGenerator
from app.schemas.predictions import (
    AnalysisResult,
    ForensicsReport,
    ModelPrediction,
)


class AnalysisService:
    """
    ImageTrust's central analysis engine.

    It coordinates the models and turns their raw outputs into one safe,
    user-facing conclusion.
    """

    def __init__(
        self,
        detector_checkpoint_path: str,
        attributor_checkpoint_path: str,
        ai_threshold: float = 0.70,
        source_threshold: float = 0.60,
    ):
        self.detector = RealAIDetector(detector_checkpoint_path)
        self.attributor = GeneratorAttributor(attributor_checkpoint_path)
        self.forensics = ImageForensicsAnalyzer()

        # Starting thresholds. We will tune these using evaluation data later.
        self.ai_threshold = ai_threshold
        self.source_threshold = source_threshold
        self.llm = ReportGenerator()

        self.graph = build_analysis_graph(detector=self.detector, 
                                          attributor=self.attributor, 
                                          forensics=self.forensics, 
                                          llm=self.llm, 
                                          ai_threshold=self.ai_threshold,
                                          source_threshold=self.source_threshold)

    def analyze(self, image_path: str) -> AnalysisResult:
       final_state = self.graph.invoke({"image_path": image_path})
       return final_state["result"]