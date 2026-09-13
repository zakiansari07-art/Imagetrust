
import mimetypes
from pathlib import Path
import markdown
import gradio as gr
import requests


API_URL = "http://127.0.0.1:7860/analyze"


# ================================================================
# API / ANALYSIS
# ================================================================
def markdown_to_html(text: str) -> str:
    return markdown.markdown(
        text,
        extensions=[
            "extra",
            "sane_lists",
        ],
    )
def analyze_uploaded_image(image_path: str):

    if not image_path:
        return (
            """
            <div class="empty-state">
                <div class="empty-icon">↑</div>
                <h3>No image selected</h3>
                <p>Upload an image to begin the ImageTrust analysis.</p>
            </div>
            """,
            {},
            {},
        )

    try:
        content_type = (
            mimetypes.guess_type(image_path)[0]
            or "application/octet-stream"
        )

        with open(image_path, "rb") as image_file:

            response = requests.post(
                API_URL,
                files={
                    "file": (
                        Path(image_path).name,
                        image_file,
                        content_type,
                    )
                },
                timeout=180,
            )

        response.raise_for_status()

        result = response.json()

    except requests.RequestException as error:

        return (
            f"""
            <div class="error-card">
                <div class="error-title">Analysis unavailable</div>
                <div class="error-message">{error}</div>
            </div>
            """,
            {},
            {},
        )

    # ============================================================
    # VERDICT
    # ============================================================

    verdict_key = result.get("verdict", "uncertain")

    verdict_labels = {
        "likely_ai_generated": "Likely AI-Generated",
        "likely_real": "Likely Real",
        
    }

    verdict = verdict_labels.get(
        verdict_key,
        verdict_key.replace("_", " ").title(),
    )

    confidence = result.get("confidence")

    confidence_text = (
        f"{confidence:.1%}"
        if confidence is not None
        else "N/A"
    )

    explanation = result.get(
        "explanation",
        "No explanation was provided.",
    )

    # ============================================================
    # VERDICT STYLE
    # ============================================================

    if verdict_key == "likely_ai_generated":
        verdict_class = "verdict-ai"
        verdict_icon = "AI"
        verdict_subtitle = "Image characteristics are more consistent with AI generation."

    else:
        verdict_key == "likely_real"
        verdict_class = "verdict-real"
        verdict_icon = "✓"
        verdict_subtitle = "Image characteristics are more consistent with a real image."

    

    # ============================================================
    # SOURCE
    # ============================================================

    condfidence_level = result.get("confidence_level")
    generator = result.get("likely_generator")
    generator_confidence = result.get(
        "generator_confidence"
    )

    if (
        verdict_key == "likely_ai_generated"
        
        and generator
    ):

        source_html = f"""
        <div class="metric-card">
            <div class="metric-label">LIKELY GENERATOR</div>
            <div class="metric-value">{generator}</div>
            <div class="metric-detail">
                Attribution confidence:
                {generator_confidence:.1%}
            </div>
        </div>
        """

    else:
        verdict_key == "likely_ai_generated"

        source_html = """
        <div class="metric-card">
            <div class="metric-label">GENERATOR</div>
            <div class="metric-value muted">Unknown</div>
            <div class="metric-detail">
                No supported generator matched with sufficient confidence.
            </div>
        </div>
        """

    

    # ============================================================
    # MODEL RESULTS
    # ============================================================

    real_ai = result.get(
        "real_vs_ai",
        {},
    )

    attribution = result.get(
        "generator_attribution",
        {},
    )

    detector_label = (
        real_ai.get("label", "Unknown")
        .replace("_", " ")
        .title()
    )

    detector_confidence = real_ai.get(
        "confidence"
    )

    detector_confidence_text = (
        f"{detector_confidence:.1%}"
        if detector_confidence is not None
        else "N/A"
    )

    attribution_label = (
        attribution.get("label", "Unknown")
        .replace("_", " ")
        .title()
    )

    attribution_confidence = attribution.get(
        "confidence"
    )

    attribution_confidence_text = (
        f"{attribution_confidence:.1%}"
        if attribution_confidence is not None
        else "N/A"
    )

    model_results_html = f"""
    <div class="section-grid">

        <div class="analysis-card">

            <div class="card-header">
                <div>
                    <div class="card-title">Real vs AI</div>
                    <div class="card-description">
                        Primary image classification
                    </div>
                </div>
            </div>

            <div class="analysis-result">
                <span class="result-label">Prediction</span>
                <span class="result-value">
                    {detector_label}
                </span>
            </div>

            <div class="analysis-result">
                <span class="result-label">Confidence</span>
                <span class="result-value">
                    {detector_confidence_text}
                </span>
            </div>

        </div>


        <div class="analysis-card">

            <div class="card-header">
                <div>
                    <div class="card-title">Generator Attribution</div>
                    <div class="card-description">
                        Supported-source similarity
                    </div>
                </div>
            </div>

            <div class="analysis-result">
                <span class="result-label">Prediction</span>
                <span class="result-value">
                    {attribution_label}
                </span>
            </div>

            <div class="analysis-result">
                <span class="result-label">Confidence</span>
                <span class="result-value">
                    {attribution_confidence_text}
                </span>
            </div>

        </div>

    </div>
    """

    # ============================================================
    # LLM REPORT
    # ============================================================

    llm_report = result.get("llm_report")

    if llm_report:

        report_html_content = markdown_to_html(
        llm_report
    )

        report_html = f"""
    <div class="report-card">

        <div class="report-header">

            <div class="report-icon">
                IT
            </div>

            <div>
                <div class="report-title">
                    ImageTrust Assessment
                </div>

                <div class="report-subtitle">
                    Automated evidence summary
                </div>
            </div>

        </div>

        <div class="report-content">
            {report_html_content}
        </div>

    </div>
    """

    else:

        report_html = f"""
        <div class="report-card">

            <div class="report-header">
                <div class="report-icon">IT</div>

                <div>
                    <div class="report-title">
                        ImageTrust Assessment
                    </div>

                    <div class="report-subtitle">
                        Automated evidence summary
                    </div>
                </div>
            </div>

            <div class="report-content">
                <p>{explanation}</p>
            </div>

        </div>
        """

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    summary_html = f"""
    <div class="verdict-card {verdict_class}">

        <div class="verdict-left">

            <div class="verdict-icon">
                {verdict_icon}
            </div>

            <div>

                <div class="verdict-eyebrow">
                    IMAGETRUST ASSESSMENT
                </div>

                <div class="verdict-title">
                    {verdict}
                </div>

                <div class="verdict-subtitle">
                    {verdict_subtitle}
                </div>

            </div>

        </div>


        <div class="confidence-box">

            <div class="confidence-label">
                CONFIDENCE
            </div>

            <div class="confidence-value">
                {confidence_text}
            </div>

        </div>

    </div>


    <div class="metrics-row">

        <div class="metric-card">

            <div class="metric-label">
                CLASSIFICATION
            </div>

            <div class="metric-value">
                {detector_label}
            </div>

            <div class="metric-detail">
                Detector confidence: {detector_confidence_text}
            </div>

        </div>

        {source_html}

    </div>


    {model_results_html}

    {report_html}
    """

    return (
        summary_html,
        {
            "real_vs_ai": real_ai,
            "generator_attribution": attribution,
        },
        result.get("forensics", {}),
    )


# ================================================================
# CUSTOM CSS
# ================================================================

CSS = """

/* ============================================================
   GLOBAL
   ============================================================ */

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif !important;
}

body {
    background: #f6f7f9 !important;
}


/* ============================================================
   HEADER
   ============================================================ */

.brand-header {
    padding: 34px 0 26px 0;
}

.brand-row {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-logo {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: #111827;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 15px;
    letter-spacing: -0.5px;
}

.brand-name {
    font-size: 28px;
    font-weight: 750;
    letter-spacing: -1px;
    color: #ffffff;
}

.brand-tagline {
    margin-top: 7px;
    color: #6b7280;
    font-size: 14px;
}


/* ============================================================
   UPLOAD CARD
   ============================================================ */

.upload-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
}

.upload-title {
    font-size: 15px;
    font-weight: 650;
    color: #111827;
    margin-bottom: 4px;
}

.upload-description {
    font-size: 13px;
    color: #6b7280;
    margin-bottom: 14px;
}


/* ============================================================
   BUTTON
   ============================================================ */

.analyze-button {
    min-height: 48px !important;
    border-radius: 11px !important;
    font-weight: 650 !important;
    font-size: 14px !important;
}


/* ============================================================
   VERDICT
   ============================================================ */

.verdict-card {
    margin-top: 24px;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    background: white;

    display: flex;
    justify-content: space-between;
    align-items: center;

    box-shadow: 0 5px 24px rgba(0, 0, 0, 0.04);
}

.verdict-left {
    display: flex;
    align-items: center;
    gap: 15px;
}

.verdict-icon {
    width: 48px;
    height: 48px;
    border-radius: 13px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 16px;
    font-weight: 800;
}

.verdict-ai .verdict-icon {
    background: #fef2f2;
    color: #dc2626;
}

.verdict-real .verdict-icon {
    background: #f0fdf4;
    color: #16a34a;
}

.verdict-uncertain .verdict-icon {
    background: #fffbeb;
    color: #d97706;
}

.verdict-eyebrow {
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 1px;
    color: #9ca3af;
    margin-bottom: 3px;
}

.verdict-title {
    font-size: 23px;
    font-weight: 750;
    letter-spacing: -0.6px;
    color: #111827;
}

.verdict-subtitle {
    margin-top: 4px;
    font-size: 13px;
    color: #6b7280;
}

.confidence-box {
    text-align: right;
}

.confidence-label {
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 1px;
    color: #9ca3af;
}

.confidence-value {
    font-size: 25px;
    font-weight: 750;
    color: #111827;
    margin-top: 3px;
}


/* ============================================================
   METRICS
   ============================================================ */

.metrics-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-top: 14px;
}

.metric-card,
.analysis-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 18px;
}

.metric-label {
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 1px;
    color: #9ca3af;
}

.metric-value {
    margin-top: 7px;
    font-size: 18px;
    font-weight: 700;
    color: #111827;
}

.metric-value.muted {
    color: #6b7280;
}

.metric-detail {
    margin-top: 4px;
    color: #6b7280;
    font-size: 12px;
    line-height: 1.5;
}


/* ============================================================
   MODEL ANALYSIS
   ============================================================ */

.section-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-top: 14px;
}

.card-title {
    font-size: 14px;
    font-weight: 700;
    color: #111827;
}

.card-description {
    margin-top: 3px;
    font-size: 12px;
    color: #9ca3af;
}

.analysis-result {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 11px 0;

    border-bottom: 1px solid #f0f1f3;
}

.analysis-result:last-child {
    border-bottom: none;
}

.result-label {
    font-size: 12px;
    color: #6b7280;
}

.result-value {
    font-size: 13px;
    font-weight: 650;
    color: #111827;
}


/* ============================================================
   REPORT
   ============================================================ */

.report-card {
    margin-top: 14px;
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 22px;

    box-shadow: 0 5px 24px rgba(0, 0, 0, 0.035);
}

.report-header {
    display: flex;
    align-items: center;
    gap: 11px;

    padding-bottom: 16px;
    border-bottom: 1px solid #eef0f2;
}

.report-icon {
    width: 34px;
    height: 34px;
    border-radius: 9px;

    background: #111827;
    color: white;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 10px;
    font-weight: 800;
}

.report-title {
    font-size: 14px;
    font-weight: 700;
    color: #111827;
}

.report-subtitle {
    margin-top: 2px;
    font-size: 11px;
    color: #374151;
}

.report-content h1 {
    display: none;
}

.report-content h2 {
    font-size: 14px;
    font-weight: 700;
    color: #111827;
    margin: 18px 0 6px 0;
}

.report-content h3 {
    font-size: 13px;
    font-weight: 650;
    color: #4b5563;
    margin: 12px 0 5px 0;
}

.report-content p {
    color: #374151 !important;
    margin: 5px 0 8px 0;
}

.report-content ul {
    color: #374151 !important;
    margin: 5px 0 10px 0;
    padding-left: 20px;
}

.report-content li {
    color: #374151 !important;
    margin: 3px 0;
}

.report-content strong {
    font-weight: 650;
    color: #4b5563;
}


/* ============================================================
   EMPTY / ERROR
   ============================================================ */

.empty-state {
    margin-top: 24px;
    background: white;
    border: 1px dashed #d1d5db;
    border-radius: 16px;
    padding: 42px 20px;
    text-align: center;
}

.empty-icon {
    margin: auto auto 12px auto;

    width: 42px;
    height: 42px;

    border-radius: 11px;
    background: #f3f4f6;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 20px;
    color: #6b7280;
}

.empty-state h3 {
    margin: 0;
    color: #111827;
    font-size: 15px;
}

.empty-state p {
    margin-top: 6px;
    color: #9ca3af;
    font-size: 13px;
}

.error-card {
    margin-top: 24px;
    padding: 18px;

    border-radius: 14px;

    background: #fff7f7;
    border: 1px solid #fecaca;
}

.error-title {
    font-weight: 700;
    color: #b91c1c;
}

.error-message {
    margin-top: 5px;
    font-size: 13px;
    color: #7f1d1d;
}


/* ============================================================
   TECHNICAL DATA
   ============================================================ */

.technical-note {
    margin-top: 20px;
    margin-bottom: 8px;

    font-size: 12px;
    color: #9ca3af;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 700px) {

    .verdict-card {
        flex-direction: column;
        align-items: flex-start;
        gap: 18px;
    }

    .confidence-box {
        text-align: left;
    }

    .metrics-row,
    .section-grid {
        grid-template-columns: 1fr;
    }

    .brand-name {
        font-size: 24px;
    }
}
"""


# ================================================================
# GRADIO APP
# ================================================================

with gr.Blocks(
    title="ImageTrust",
    theme=gr.themes.Base(
        primary_hue="slate",
        neutral_hue="slate",
        font=[
            gr.themes.GoogleFont("Inter"),
            "system-ui",
            "sans-serif",
        ],
    ),
    css=CSS,
) as demo:

    # ------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------

    gr.HTML(
        """
        <div class="brand-header">

            <div class="brand-row">

                <div class="brand-logo">
                    IT
                </div>

                <div>
                    <div class="brand-name">
                        ImageTrust
                    </div>

                    <div class="brand-tagline">
                        AI image detection · generator attribution · digital forensics
                    </div>
                </div>

            </div>

        </div>
        """
    )

    # ------------------------------------------------------------
    # UPLOAD
    # ------------------------------------------------------------

    with gr.Row():

        with gr.Column(
            scale=1,
            elem_classes="upload-card",
        ):

            gr.HTML(
                """
                <div class="upload-title">
                    Analyze an image
                </div>

                <div class="upload-description">
                    Upload an image to evaluate its authenticity,
                    possible AI origin, and available provenance signals.
                </div>
                """
            )

            image_input = gr.Image(
                type="filepath",
                label="",
                height=360,
            )

            analyze_button = gr.Button(
                "Analyze Image",
                variant="primary",
                elem_classes="analyze-button",
            )

    # ------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------

    report_output = gr.HTML(
        value="""
        <div class="empty-state">

            <div class="empty-icon">
                ↑
            </div>

            <h3>
                Ready for analysis
            </h3>

            <p>
                Upload an image above and run ImageTrust.
            </p>

        </div>
        """
    )

    # ------------------------------------------------------------
    # TECHNICAL DETAILS
    # ------------------------------------------------------------

    gr.HTML(
        """
        <div class="technical-note">
            Technical outputs
        </div>
        """
    )

    with gr.Accordion(
        "Model results",
        open=False,
    ):

        model_output = gr.JSON(
            label="",
        )

    with gr.Accordion(
        "Digital forensics",
        open=False,
    ):

        forensics_output = gr.JSON(
            label="",
        )

    # ------------------------------------------------------------
    # ACTION
    # ------------------------------------------------------------

    analyze_button.click(
        fn=analyze_uploaded_image,
        inputs=image_input,
        outputs=[
            report_output,
            model_output,
            forensics_output,
        ],
        show_progress="full",
    )


# ================================================================
# LAUNCH
# ================================================================

