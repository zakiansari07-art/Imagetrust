import mimetypes
from pathlib import Path

import gradio as gr
import requests


API_URL = "http://127.0.0.1:7860/analyze"


def analyze_uploaded_image(image_path: str):
    if not image_path:
        return "Please upload an image.", {}, {}

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
            f"## Analysis Unavailable\n\n"
            f"> {error}",
            {},
            {},
        )

    # ---------------------------------------------------------
    # RESULT DATA
    # ---------------------------------------------------------

    verdict = result["verdict"].replace("_", " ").title()

    confidence = result.get("confidence")
    confidence_level = result.get("confidence_level")


    explanation = result.get(
        "explanation",
        "No explanation was provided.",
    )

    likely_generator = result.get("likely_generator")
    generator_confidence = result.get("generator_confidence")
    

    llm_report = result.get("llm_report")

    # ---------------------------------------------------------
    # VERDICT
    # ---------------------------------------------------------

    if result["verdict"] == "likely_ai_generated":
        verdict_icon = "⚠️"

    else:
        result["verdict"] == "likely_real"
        verdict_icon = "✓"

   

    # ---------------------------------------------------------
    # MAIN REPORT
    # ---------------------------------------------------------

    final_report = f"""
# Analysis Result

## {verdict_icon} {verdict}

> {explanation}

---

## Analysis Summary

| Analysis | Result |
|---|---|
| **AI Detection** | **{verdict}** |
| **Detection Confidence** | **{confidence}** |
| **Confidence Level** | **{confidence_level}** |
| **Likely Generator** | **{likely_generator}** |
| **Generator Confidence** | **{generator_confidence}** |

---

{llm_report if llm_report else "_Report is unavailable._"}
"""

    # ---------------------------------------------------------
    # MODEL OUTPUT
    # ---------------------------------------------------------

    model_results = {
        "real_vs_ai": result.get("real_vs_ai"),
        "generator_attribution": result.get(
            "generator_attribution"
        ),
    }

    # ---------------------------------------------------------
    # RETURN
    # ---------------------------------------------------------

    return (
        final_report,
        model_results,
        result.get("forensics", {}),
    )


# =============================================================
# GRADIO APPLICATION
# =============================================================



with gr.Blocks(title="ImageTrust", theme=gr.themes.Default(primary_hue="emerald", secondary_hue="stone", neutral_hue="gray")) as demo:

    # ---------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------

    gr.Markdown(
        """
# ImageTrust

**AI Image Authenticity & Attribution**

Analyze an image for AI-generation signals, identify the
likely generation source when sufficient evidence exists,
and inspect technical forensic information.
"""
    )

    gr.Markdown("---")

    # ---------------------------------------------------------
    # IMAGE INPUT
    # ---------------------------------------------------------

    gr.Markdown("## Analyze an Image")

    with gr.Row(equal_height=True):

        with gr.Column():
            image_input = gr.Image(
                type="filepath",
                label="Upload Image",
            )

        with gr.Column():
            gr.Markdown(
                """
### What ImageTrust checks ?""")
            gr.Markdown("---")

            gr.Markdown("""

**Real vs AI Detection**  
Determines whether the image is likely real or AI-generated.

**Generator Attribution**  
If the image appears AI-generated, estimates the most likely
supported generator.

**Image Forensics**  
Examines technical properties and available image metadata.

**Generates Report**  
Converts the structured analysis into an understandable report.
"""
            )

    analyze_button = gr.Button(
        "Analyze Image",
        variant="primary",
    )

    gr.Markdown("---")

    # ---------------------------------------------------------
    # ANALYSIS RESULT
    # ---------------------------------------------------------

    gr.Markdown("## Analysis Result")

    report_output = gr.Markdown(
        value=(
            "Upload an image and click **Analyze Image** "
            "to begin."
        ),
        container=True,
    )

    # ---------------------------------------------------------
    # TECHNICAL DETAILS
    # ---------------------------------------------------------

    gr.Markdown("---")

    with gr.Accordion(
        "Model Analysis Json",
        open=False,
    ):
        model_output = gr.JSON(
            label="Detector and Generator Attribution"
        )

    with gr.Accordion(
        "Forensic Evaluation Json",
        open=False,
    ):
        forensics_output = gr.JSON(
            label="Technical Image Details",
        )

    # ---------------------------------------------------------
    # BUTTON EVENT
    # ---------------------------------------------------------

    analyze_button.click(
        fn=analyze_uploaded_image,
        inputs=image_input,
        outputs=[
            report_output,
            model_output,
            forensics_output,
        ],
    )


