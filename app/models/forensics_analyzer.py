import cv2
from pathlib import Path
from PIL import Image, ExifTags
import numpy as np
from app.services.analyze_provenance import ProvenanceAnalyzer

class ImageForensicsAnalyzer:
    """
    Extracts traditional digital-forensics features from an image.

    Important:
        These features are forensic indicators. They do NOT independently
        prove that an image is AI-generated or real.
    """
    

    def __init__(self):

        self.provenance = ProvenanceAnalyzer()
    def analyze(self, image_path: str):

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # ============================================================
        # 1. LOAD IMAGE + BASIC METADATA
        # ============================================================
        provenance = self.provenance.analyze(image_path)
        with Image.open(path) as image:

            image_format = image.format
            width, height = image.size
            original_mode = image.mode


            rgb_image = image.convert("RGB")

        # NumPy representation
        rgb = np.array(rgb_image)

        # OpenCV representation
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        grayscale = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # ============================================================
        # 2. BASIC IMAGE INFORMATION
        # ============================================================

        file_size = path.stat().st_size

        aspect_ratio = width / height if height != 0 else 0

        # ============================================================
        # 3. COLOR ANALYSIS
        # ============================================================

        mean_rgb = rgb.mean(axis=(0, 1))

        std_rgb = rgb.std(axis=(0, 1))

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        mean_saturation = float(hsv[:, :, 1].mean())

        mean_brightness = float(hsv[:, :, 2].mean())

        # ============================================================
        # 4. QUALITY ANALYSIS
        # ============================================================

        # Variance of Laplacian is a common blur/sharpness indicator.
        laplacian = cv2.Laplacian(
            grayscale,
            cv2.CV_64F
        )

        sharpness_score = float(laplacian.var())

        # Blur score:
        # Higher value generally means sharper image.
        blur_score = sharpness_score

        # Contrast
        contrast = float(grayscale.std())

        # Histogram entropy
        entropy = self._calculate_entropy(grayscale)

        # ============================================================
        # 5. NOISE ANALYSIS
        # ============================================================

        noise = self._extract_noise(grayscale)

        noise_mean = float(np.mean(noise))
        noise_std = float(np.std(noise))

        noise_median = float(np.median(noise))

        noise_mad = float(
            np.median(np.abs(noise - np.median(noise)))
        )

        # Spatial noise variation
        noise_spatial_std = float(
            np.std(np.abs(noise))
        )

        # ============================================================
        # 6. EDGE ANALYSIS
        # ============================================================

        edges = cv2.Canny(
            grayscale,
            threshold1=100,
            threshold2=200
        )

        edge_density = float(
            np.count_nonzero(edges) / edges.size
        )

        edge_mean = float(edges.mean())

        # ============================================================
        # 7. FREQUENCY / DCT ANALYSIS
        # ============================================================

        frequency = self._frequency_analysis(grayscale)

        # ============================================================
        # 8. JPEG / COMPRESSION ANALYSIS
        # ============================================================

        compression = self._compression_analysis(
            path,
            rgb_image,
            grayscale
        )

        # ============================================================
        # 9. ERROR LEVEL ANALYSIS
        # ============================================================

        ela = self._error_level_analysis(
            path,
            rgb_image
        )

        # ============================================================
        # 10. RESAMPLING / UPSCALING ANALYSIS
        # ============================================================

        resampling = self._resampling_analysis(
            grayscale
        )

        # ============================================================
        # 11. DOUBLE COMPRESSION INDICATORS
        # ============================================================

        double_compression = self._double_compression_analysis(
            path,
            grayscale,
            compression
        )

        # ============================================================
        # 12. IMAGE DIMENSION / UPSCALING INDICATORS
        # ============================================================

        resizing = self._resizing_analysis(
            width,
            height,
            grayscale
        )

        # ============================================================
        # 13. RETURN COMPLETE FORENSIC REPORT
        # ============================================================

        return {
            "file": {
                "file_name": path.name,
                "file_size_bytes": file_size,
                "file_format": image_format
            },

            "image": {
                "width": width,
                "height": height,
                "channels": rgb.shape[2],
                "aspect_ratio": round(float(aspect_ratio), 4)
            },

            "metadata": provenance
            ,

            "color": {
                "mean_rgb": [
                    round(float(x), 2)
                    for x in mean_rgb
                ],

                "std_rgb": [
                    round(float(x), 2)
                    for x in std_rgb
                ],

                "mean_saturation": round(
                    mean_saturation,
                    2
                ),

                "mean_brightness": round(
                    mean_brightness,
                    2
                )
            },

            "quality": {
                "sharpness_score": round(
                    sharpness_score,
                    2
                ),

                "blur_score": round(
                    blur_score,
                    2
                ),

                "contrast": round(
                    contrast,
                    2
                ),

                "entropy": round(
                    entropy,
                    4
                )
            },

            "noise": {
                "noise_mean": round(
                    noise_mean,
                    4
                ),

                "noise_std": round(
                    noise_std,
                    4
                ),

                "noise_median": round(
                    noise_median,
                    4
                ),

                "noise_mad": round(
                    noise_mad,
                    4
                ),

                "noise_spatial_std": round(
                    noise_spatial_std,
                    4
                )
            },

            "edges": {
                "edge_density": round(
                    edge_density,
                    6
                ),

                "edge_mean": round(
                    edge_mean,
                    4
                )
            },

            "frequency": frequency,

            "compression": compression,

            "ela": ela,

            "resampling": resampling,

            "double_compression": double_compression,

            "resizing": resizing,

            "provenance": provenance
        }

    # ================================================================
    # ENTROPY
    # ================================================================

    @staticmethod
    def _calculate_entropy(image):

        histogram = cv2.calcHist(
            [image],
            [0],
            None,
            [256],
            [0, 256]
        )

        histogram = histogram.flatten()

        probability = histogram / np.sum(histogram)

        probability = probability[
            probability > 0
        ]

        entropy = -np.sum(
            probability * np.log2(probability)
        )

        return float(entropy)

    # ================================================================
    # NOISE EXTRACTION
    # ================================================================

    @staticmethod
    def _extract_noise(grayscale):

        """
        Estimate high-frequency noise by subtracting a Gaussian-smoothed
        image from the original.

        This is NOT a camera PRNU measurement.
        """

        blurred = cv2.GaussianBlur(
            grayscale,
            (5, 5),
            0
        )

        noise = (
            grayscale.astype(np.float32)
            - blurred.astype(np.float32)
        )

        return noise

    # ================================================================
    # FREQUENCY ANALYSIS
    # ================================================================

    @staticmethod
    def _frequency_analysis(grayscale):

        image = grayscale.astype(
            np.float32
        )

        # Remove average intensity
        image -= image.mean()

        fft = np.fft.fft2(image)

        fft_shift = np.fft.fftshift(fft)

        magnitude = np.abs(fft_shift)

        energy = magnitude ** 2

        h, w = grayscale.shape

        cy = h // 2
        cx = w // 2

        # Radius of low-frequency region
        radius = min(h, w) * 0.10

        y, x = np.ogrid[:h, :w]

        distance = np.sqrt(
            (x - cx) ** 2 +
            (y - cy) ** 2
        )

        low_mask = distance <= radius

        high_mask = distance > radius

        low_frequency_energy = float(
            energy[low_mask].mean()
        )

        high_frequency_energy = float(
            energy[high_mask].mean()
        )

        total_energy = float(
            energy.mean()
        )

        high_frequency_ratio = (
            high_frequency_energy /
            (total_energy + 1e-12)
        )

        return {
            "low_frequency_energy": round(
                low_frequency_energy,
                4
            ),

            "high_frequency_energy": round(
                high_frequency_energy,
                4
            ),

            "high_frequency_ratio": round(
                float(high_frequency_ratio),
                6
            )
        }

    # ================================================================
    # JPEG / COMPRESSION ANALYSIS
    # ================================================================

    def _compression_analysis(
        self,
        path,
        image,
        grayscale
    ):

        suffix = path.suffix.lower()

        jpeg_detected = suffix in {
            ".jpg",
            ".jpeg"
        }

        quality_estimate = None
        quantization_tables = None

        # Pillow exposes JPEG quantization tables.
        with Image.open(path) as img:

            if hasattr(img, "quantization"):

                quantization_tables = img.quantization

                if quantization_tables:

                    quality_estimate = (
                        self._estimate_jpeg_quality(
                            quantization_tables
                        )
                    )

        # JPEG 8x8 block artifact analysis
        block_artifact_score = (
            self._jpeg_block_artifact_score(
                grayscale
            )
        )

        # DCT coefficient statistics
        dct_statistics = (
            self._dct_block_statistics(
                grayscale
            )
        )

        return {
            "jpeg_detected": jpeg_detected,

            "estimated_jpeg_quality": (
                quality_estimate
            ),

            "block_artifact_score": round(
                block_artifact_score,
                6
            ),

            "dct": dct_statistics
        }

    # ================================================================
    # JPEG QUALITY ESTIMATION
    # ================================================================

    @staticmethod
    def _estimate_jpeg_quality(
        quantization_tables
    ):

        """
        Rough JPEG quality estimate.

        JPEG encoders differ, so this should be treated as an estimate,
        not an exact original quality value.
        """

        table = quantization_tables.get(0)

        if table is None:
            return None

        standard_luminance = np.array(
            [
                16, 11, 10, 16, 24, 40, 51, 61,
                12, 12, 14, 19, 26, 58, 60, 55,
                14, 13, 16, 24, 40, 57, 69, 56,
                14, 17, 22, 29, 51, 87, 80, 62,
                18, 22, 37, 56, 68, 109, 103, 77,
                24, 35, 55, 64, 81, 104, 113, 92,
                49, 64, 78, 87, 103, 121, 120, 101,
                72, 92, 95, 98, 112, 100, 103, 99
            ]
        )

        table = np.array(table)

        scale = np.mean(
            table / standard_luminance
        )

        if scale <= 1:
            quality = 100

        elif scale < 50:
            quality = int(
                round(
                    100 - scale * 1.5
                )
            )

        else:
            quality = int(
                round(
                    5000 / scale
                )
            )

        return max(
            1,
            min(
                100,
                quality
            )
        )

    # ================================================================
    # JPEG BLOCK ARTIFACT DETECTION
    # ================================================================

    @staticmethod
    def _jpeg_block_artifact_score(grayscale):

        image = grayscale.astype(
            np.float32
        )

        h, w = image.shape

        vertical_differences = []

        horizontal_differences = []

        # Differences across 8-pixel JPEG block boundaries
        for x in range(8, w, 8):

            diff = np.abs(
                image[:, x] -
                image[:, x - 1]
            )

            vertical_differences.append(
                np.mean(diff)
            )

        for y in range(8, h, 8):

            diff = np.abs(
                image[y, :] -
                image[y - 1, :]
            )

            horizontal_differences.append(
                np.mean(diff)
            )

        if not vertical_differences:
            return 0.0

        boundary_energy = np.mean(
            vertical_differences +
            horizontal_differences
        )

        # Compare against neighboring non-boundary positions
        non_boundary = []

        for x in range(1, w):

            if x % 8 != 0:

                non_boundary.append(
                    np.mean(
                        np.abs(
                            image[:, x] -
                            image[:, x - 1]
                        )
                    )
                )

        if not non_boundary:
            return 0.0

        normal_energy = np.mean(
            non_boundary
        )

        score = (
            boundary_energy /
            (normal_energy + 1e-8)
        )

        return float(score)

    # ================================================================
    # DCT BLOCK ANALYSIS
    # ================================================================

    @staticmethod
    def _dct_block_statistics(grayscale):

        image = grayscale.astype(
            np.float32
        )

        h, w = image.shape

        h -= h % 8
        w -= w % 8

        image = image[:h, :w]

        low_energy = []
        high_energy = []

        for y in range(0, h, 8):

            for x in range(0, w, 8):

                block = image[
                    y:y + 8,
                    x:x + 8
                ]

                block -= block.mean()

                dct = cv2.dct(block)

                energy = dct ** 2

                # Low frequency coefficients
                low = energy[:3, :3].sum()

                # High frequency coefficients
                high = energy[4:, 4:].sum()

                low_energy.append(low)
                high_energy.append(high)

        if not high_energy:
            return {
                "mean_low_frequency_dct_energy": 0.0,
                "mean_high_frequency_dct_energy": 0.0,
                "high_frequency_dct_ratio": 0.0
            }

        mean_low = np.mean(low_energy)
        mean_high = np.mean(high_energy)

        ratio = (
            mean_high /
            (mean_low + 1e-8)
        )

        return {
            "mean_low_frequency_dct_energy": round(
                float(mean_low),
                4
            ),

            "mean_high_frequency_dct_energy": round(
                float(mean_high),
                4
            ),

            "high_frequency_dct_ratio": round(
                float(ratio),
                6
            )
        }

    # ================================================================
    # ERROR LEVEL ANALYSIS
    # ================================================================

    @staticmethod
    def _error_level_analysis(
        path,
        image
    ):

        """
        Re-compress image at a known JPEG quality and compare
        the original against the recompressed image.

        ELA is primarily meaningful for JPEG images.
        """

        if path.suffix.lower() not in {
            ".jpg",
            ".jpeg"
        }:

            return {
                "applicable": False,
                "mean_error": None,
                "std_error": None,
                "max_error": None
            }

        try:

            import io

            buffer = io.BytesIO()

            image.save(
                buffer,
                format="JPEG",
                quality=90
            )

            buffer.seek(0)

            recompressed = np.array(
                Image.open(buffer)
                .convert("RGB")
            ).astype(np.float32)

            original = np.array(
                image
            ).astype(np.float32)

            difference = np.abs(
                original -
                recompressed
            )

            mean_error = difference.mean()

            std_error = difference.std()

            max_error = difference.max()

            return {
                "applicable": True,

                "mean_error": round(
                    float(mean_error),
                    4
                ),

                "std_error": round(
                    float(std_error),
                    4
                ),

                "max_error": round(
                    float(max_error),
                    4
                )
            }

        except Exception as e:

            return {
                "applicable": False,
                "error": str(e)
            }

    # ================================================================
    # RESAMPLING ANALYSIS
    # ================================================================

    @staticmethod
    def _resampling_analysis(grayscale):

        """
        Estimates whether the image contains characteristics consistent
        with interpolation/resampling.

        This does NOT prove that an image was resized.
        """

        image = grayscale.astype(
            np.float32
        )

        h, w = image.shape

        if h < 100 or w < 100:

            return {
                "resampling_score": None,
                "interpolation_error": None
            }

        # Downsample then reconstruct
        small = cv2.resize(
            image,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA
        )

        reconstructed = cv2.resize(
            small,
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )

        error = np.abs(
            image -
            reconstructed
        )

        interpolation_error = float(
            error.mean()
        )

        # Analyze periodicity of interpolation residual
        residual = (
            image -
            cv2.GaussianBlur(
                image,
                (3, 3),
                0
            )
        )

        spectrum = np.abs(
            np.fft.fftshift(
                np.fft.fft2(residual)
            )
        )

        spectrum = spectrum / (
            spectrum.mean() + 1e-8
        )

        # Strong periodic frequency peaks
        percentile = np.percentile(
            spectrum,
            99.5
        )

        resampling_score = (
            percentile /
            (spectrum.mean() + 1e-8)
        )

        return {
            "resampling_score": round(
                float(resampling_score),
                4
            ),

            "interpolation_error": round(
                interpolation_error,
                4
            )
        }

    # ================================================================
    # DOUBLE COMPRESSION ANALYSIS
    # ================================================================

    @staticmethod
    def _double_compression_analysis(
        path,
        grayscale,
        compression
    ):

        """
        Looks for indicators associated with repeated JPEG processing.

        This is a heuristic, not a forensic proof of double compression.
        """

        if not compression["jpeg_detected"]:

            return {
                "applicable": False,
                "double_compression_indicator": False,
                "score": None
            }

        block_score = (
            compression["block_artifact_score"]
        )

        quality = (
            compression["estimated_jpeg_quality"]
        )

        if quality is None:

            return {
                "applicable": True,
                "double_compression_indicator": False,
                "score": None
            }

        # Strong block artifacts + moderate/low JPEG quality
        # can be consistent with repeated compression.
        score = (
            block_score *
            (100 - quality + 1)
        )

        indicator = (
            block_score > 1.10
            and quality < 90
        )

        return {
            "applicable": True,

            "double_compression_indicator": bool(
                indicator
            ),

            "score": round(
                float(score),
                4
            )
        }

    # ================================================================
    # RESIZING / UPSCALING ANALYSIS
    # ================================================================

    @staticmethod
    def _resizing_analysis(
        width,
        height,
        grayscale
    ):

        """
        Looks for dimensions and interpolation characteristics that can
        be associated with resizing/upscaling.

        Without the original image, this cannot establish that resizing
        actually occurred.
        """

        common_dimensions = {
            256,
            384,
            512,
            640,
            768,
            1024,
            1280,
            1536,
            1920,
            2048,
            2560,
            3072,
            3840,
            4096
        }

        dimension_match = (
            width in common_dimensions
            or height in common_dimensions
        )

        # Edge preservation after down/up sampling
        image = grayscale.astype(
            np.float32
        )

        reduced = cv2.resize(
            image,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA
        )

        restored = cv2.resize(
            reduced,
            (width, height),
            interpolation=cv2.INTER_CUBIC
        )

        reconstruction_error = float(
            np.mean(
                np.abs(
                    image -
                    restored
                )
            )
        )

        return {
            "common_dimension_detected": bool(
                dimension_match
            ),

            "reconstruction_error": round(
                reconstruction_error,
                4
            )
        }