import mimetypes
from pathlib import Path

import gradio as gr
import requests


API_URL = "http://127.0.0.1:8000/analyze"


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
        return f"## Analysis unavailable\n\n{error}", {}, {}

    verdict = result["verdict"].replace("_", " ").title()
    confidence = result["confidence"]

    summary = f"## {verdict}\n\n{result['explanation']}"

    if confidence is not None:
        summary += f"\n\n**Detection confidence:** {confidence:.1%}"

    if result["verdict"] == "likely_ai_generated":
        if result["source_status"] == "identified":
            summary += (
                f"\n\n**Likely source:** {result['likely_generator']} "
                f"({result['generator_confidence']:.1%})"
            )
        else:
            summary += (
                "\n\n**Likely source:** Unknown or unsupported generator"
            )

    llm_report = result.get("llm_report")

    if llm_report:
        summary += f"\n\n---\n\n## Plain-English report\n\n{llm_report}"
    else:
        summary += (
            "\n\n---\n\n*The structured analysis completed, but the "
            "plain-English report was unavailable.*"
        )

    model_results = {
        "real_vs_ai": result["real_vs_ai"],
        "generator_attribution": result["generator_attribution"],
    }

    return summary, model_results, result["forensics"]


with gr.Blocks(title="ImageTrust") as demo:
    gr.Markdown(
        "# ImageTrust\n"
        "Upload an image to check whether it is likely real or AI-generated."
    )

    with gr.Row(equal_height=True):
        image_input = gr.Image(
            type="filepath",
            label="Upload image",
        )

        

        report_output = gr.Textbox(label="Report", interactive=False,)
                            
    with gr.Row():             
        analyze_button = gr.Button(
            "Analyze",
            variant="primary",
        )

            

    with gr.Row():
        model_output = gr.JSON(label="Model results")
        forensics_output = gr.JSON(
            label="Technical image details",
        )

    analyze_button.click(
        fn=analyze_uploaded_image,
        inputs=image_input,
        outputs=[
            report_output,
            model_output,
            forensics_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()