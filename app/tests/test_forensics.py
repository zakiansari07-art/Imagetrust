import json
import sys
from pathlib import Path

from app.models.forensics_analyzer import ImageForensicsAnalyzer


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():

    # ------------------------------------------------------------
    # Get image path
    # ------------------------------------------------------------

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "python test_forensics.py <image_path>"
        )

        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():

        print(
            f"ERROR: Image not found: {image_path}"
        )

        sys.exit(1)

    # ------------------------------------------------------------
    # Create analyzer
    # ------------------------------------------------------------

    analyzer = ImageForensicsAnalyzer()

    # ------------------------------------------------------------
    # Run analysis
    # ------------------------------------------------------------

    print(
        f"\nAnalyzing: {image_path}"
    )

    try:

        result = analyzer.analyze(
            str(image_path)
        )

    except Exception as e:

        print(
            f"\nERROR during analysis:\n{e}"
        )

        sys.exit(1)

    # ------------------------------------------------------------
    # Basic information
    # ------------------------------------------------------------

    print_section("IMAGE")

    image_info = result["image"]

    print(
        f"Dimensions: "
        f"{image_info['width']} x "
        f"{image_info['height']}"
    )

    print(
        f"Channels: "
        f"{image_info['channels']}"
    )

    print(
        f"Aspect ratio: "
        f"{image_info['aspect_ratio']}"
    )

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    print_section("METADATA")

    metadata = result["metadata"]

    print(
        f"Color mode: "
        f"{metadata['original_color_mode']}"
    )

    print(
        f"EXIF present: "
        f"{metadata['has_exif_metadata']}"
    )

    # ------------------------------------------------------------
    # Color
    # ------------------------------------------------------------

    print_section("COLOR")

    color = result["color"]

    print(
        f"Mean RGB: "
        f"{color['mean_rgb']}"
    )

    print(
        f"RGB standard deviation: "
        f"{color['std_rgb']}"
    )

    print(
        f"Mean saturation: "
        f"{color['mean_saturation']}"
    )

    print(
        f"Mean brightness: "
        f"{color['mean_brightness']}"
    )

    # ------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------

    print_section("QUALITY")

    quality = result["quality"]

    print(
        f"Sharpness: "
        f"{quality['sharpness_score']}"
    )

    print(
        f"Blur score: "
        f"{quality['blur_score']}"
    )

    print(
        f"Contrast: "
        f"{quality['contrast']}"
    )

    print(
        f"Entropy: "
        f"{quality['entropy']}"
    )

    # ------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------

    print_section("NOISE")

    noise = result["noise"]

    print(
        f"Noise mean: "
        f"{noise['noise_mean']}"
    )

    print(
        f"Noise std: "
        f"{noise['noise_std']}"
    )

    print(
        f"Noise median: "
        f"{noise['noise_median']}"
    )

    print(
        f"Noise MAD: "
        f"{noise['noise_mad']}"
    )

    print(
        f"Noise spatial std: "
        f"{noise['noise_spatial_std']}"
    )

    # ------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------

    print_section("EDGES")

    edges = result["edges"]

    print(
        f"Edge density: "
        f"{edges['edge_density']}"
    )

    print(
        f"Edge mean: "
        f"{edges['edge_mean']}"
    )

    # ------------------------------------------------------------
    # Frequency
    # ------------------------------------------------------------

    print_section("FREQUENCY")

    frequency = result["frequency"]

    print(
        f"Low frequency energy: "
        f"{frequency['low_frequency_energy']}"
    )

    print(
        f"High frequency energy: "
        f"{frequency['high_frequency_energy']}"
    )

    print(
        f"High frequency ratio: "
        f"{frequency['high_frequency_ratio']}"
    )

    # ------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------

    print_section("COMPRESSION")

    compression = result["compression"]

    print(
        f"JPEG detected: "
        f"{compression['jpeg_detected']}"
    )

    print(
        f"Estimated JPEG quality: "
        f"{compression['estimated_jpeg_quality']}"
    )

    print(
        f"Block artifact score: "
        f"{compression['block_artifact_score']}"
    )

    print(
        "DCT statistics:"
    )

    print(
        json.dumps(
            compression["dct"],
            indent=4
        )
    )

    # ------------------------------------------------------------
    # ELA
    # ------------------------------------------------------------

    print_section("ERROR LEVEL ANALYSIS")

    ela = result["ela"]

    print(
        f"Applicable: "
        f"{ela.get('applicable')}"
    )

    print(
        f"Mean error: "
        f"{ela.get('mean_error')}"
    )

    print(
        f"Std error: "
        f"{ela.get('std_error')}"
    )

    print(
        f"Max error: "
        f"{ela.get('max_error')}"
    )

    # ------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------

    print_section("RESAMPLING")

    resampling = result["resampling"]

    print(
        f"Resampling score: "
        f"{resampling.get('resampling_score')}"
    )

    print(
        f"Interpolation error: "
        f"{resampling.get('interpolation_error')}"
    )

    # ------------------------------------------------------------
    # Double compression
    # ------------------------------------------------------------

    print_section("DOUBLE COMPRESSION")

    double_compression = result[
        "double_compression"
    ]

    print(
        f"Applicable: "
        f"{double_compression.get('applicable')}"
    )

    print(
        f"Indicator: "
        f"{double_compression.get('double_compression_indicator')}"
    )

    print(
        f"Score: "
        f"{double_compression.get('score')}"
    )

    # ------------------------------------------------------------
    # Resizing
    # ------------------------------------------------------------

    print_section("RESIZING")

    resizing = result["resizing"]

    print(
        f"Common dimension detected: "
        f"{resizing.get('common_dimension_detected')}"
    )

    print(
        f"Reconstruction error: "
        f"{resizing.get('reconstruction_error')}"
    )

    # ============================================================
    # PROVENANCE
    # ============================================================

    print_section("PROVENANCE")

    provenance = result["provenance"]

    # ------------------------------------------------------------
    # EXIF
    # ------------------------------------------------------------

    print("\nEXIF")

    exif = provenance["exif"]

    print(
        f"Present: "
        f"{exif.get('present')}"
    )

    if exif.get("camera"):

        print(
            f"Camera make: "
            f"{exif['camera'].get('make')}"
        )

        print(
            f"Camera model: "
            f"{exif['camera'].get('model')}"
        )

        print(
            f"Software: "
            f"{exif['camera'].get('software')}"
        )

        print(
            f"Date/time: "
            f"{exif['camera'].get('date_time')}"
        )

    # ------------------------------------------------------------
    # XMP
    # ------------------------------------------------------------

    print("\nXMP")

    xmp = provenance["xmp"]

    print(
        f"Present: "
        f"{xmp.get('present')}"
    )

    if xmp.get("fields"):

        print(
            json.dumps(
                xmp["fields"],
                indent=4
            )
        )

    # ------------------------------------------------------------
    # C2PA
    # ------------------------------------------------------------

    print("\nC2PA")

    c2pa = provenance["c2pa"]

    print(
        f"Library available: "
        f"{c2pa.get('available')}"
    )

    print(
        f"Manifest present: "
        f"{c2pa.get('present')}"
    )

    print(
        f"Verified: "
        f"{c2pa.get('verified')}"
    )

    if c2pa.get("error"):

        print(
            f"C2PA message: "
            f"{c2pa['error']}"
        )

    # ------------------------------------------------------------
    # C2PA manifest information
    # ------------------------------------------------------------

    manifests = c2pa.get(
        "manifests",
        []
    )

    if manifests:

        print(
            f"\nC2PA manifests found: "
            f"{len(manifests)}"
        )

        for index, manifest in enumerate(
            manifests,
            start=1
        ):

            print(
                f"\nManifest {index}"
            )

            print(
                f"  ID: "
                f"{manifest.get('manifest_id')}"
            )

            print(
                f"  Title: "
                f"{manifest.get('title')}"
            )

            print(
                f"  Format: "
                f"{manifest.get('format')}"
            )

            print(
                f"  Claim generator: "
                f"{manifest.get('claim_generator')}"
            )

            signature = manifest.get(
                "signature",
                {}
            )

            print(
                f"  Signature issuer: "
                f"{signature.get('issuer')}"
            )

            print(
                f"  Signature time: "
                f"{signature.get('time')}"
            )

            print(
                f"  Signature algorithm: "
                f"{signature.get('algorithm')}"
            )

            print(
                f"  Actions: "
                f"{len(manifest.get('actions', []))}"
            )

            print(
                f"  Ingredients: "
                f"{len(manifest.get('ingredients', []))}"
            )

    # ============================================================
    # SAVE COMPLETE JSON REPORT
    # ============================================================

    output_path = (
        image_path.parent /
        f"{image_path.stem}_forensics.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False
        )

    print_section("COMPLETE RESULT")

    print(
        f"Full JSON report saved to:\n"
        f"{output_path}"
    )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()