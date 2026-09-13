import json
import re
from pathlib import Path

from PIL import Image, ExifTags


class ProvenanceAnalyzer:
    """
    Analyze image provenance information.

    Checks:
        - EXIF
        - XMP
        - C2PA Content Credentials

    Important:
        Absence of provenance information does NOT mean that
        an image is AI-generated.
    """

    def analyze(self, image_path: str) -> dict:
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        return {
            "exif": self._analyze_exif(path),
            "xmp": self._analyze_xmp(path),
            "c2pa": self._analyze_c2pa(path),
        }

    # ================================================================
    # EXIF
    # ================================================================

    @staticmethod
    def _analyze_exif(path: Path) -> dict:

        empty_camera = {
            "make": None,
            "model": None,
            "software": None,
            "date_time": None,
        }

        try:
            with Image.open(path) as image:
                exif = image.getexif()

                if not exif:
                    return {
                        "present": False,
                        "camera": empty_camera,
                        "fields": {},
                    }

                fields = {}

                for tag_id, value in exif.items():

                    tag_name = ExifTags.TAGS.get(
                        tag_id,
                        str(tag_id),
                    )

                    if isinstance(value, bytes):
                        try:
                            value = value.decode(
                                errors="ignore"
                            )
                        except Exception:
                            value = "<binary>"

                    fields[tag_name] = str(value)

                return {
                    "present": True,

                    "camera": {
                        "make": fields.get("Make"),
                        "model": fields.get("Model"),
                        "software": fields.get("Software"),
                        "date_time": fields.get("DateTime"),
                    },

                    "fields": fields,
                }

        except Exception as e:

            return {
                "present": False,

                "camera": empty_camera,

                "fields": {},

                "error": str(e),
            }

    # ================================================================
    # XMP
    # ================================================================

    @staticmethod
    def _analyze_xmp(path: Path) -> dict:

        try:
            data = path.read_bytes()

            xmp_start = data.find(
                b"<x:xmpmeta"
            )

            if xmp_start == -1:
                xmp_start = data.find(
                    b"<rdf:RDF"
                )

            if xmp_start == -1:
                return {
                    "present": False,
                    "fields": {},
                }

            xmp_end = data.find(
                b"</x:xmpmeta>",
                xmp_start,
            )

            if xmp_end != -1:

                xmp_end += len(
                    b"</x:xmpmeta>"
                )

            else:

                xmp_end = data.find(
                    b"</rdf:RDF>",
                    xmp_start,
                )

                if xmp_end != -1:
                    xmp_end += len(
                        b"</rdf:RDF>"
                    )

            if xmp_end == -1:
                return {
                    "present": True,
                    "fields": {},
                    "raw_detected": True,
                }

            xmp_bytes = data[
                xmp_start:xmp_end
            ]

            xmp_text = xmp_bytes.decode(
                "utf-8",
                errors="ignore",
            )

            fields = {}

            patterns = {
                "creator_tool":
                    r"CreatorTool[^>]*>(.*?)</",

                "create_date":
                    r"CreateDate[^>]*>(.*?)</",

                "modify_date":
                    r"ModifyDate[^>]*>(.*?)</",

                "creator":
                    r"dc:creator[^>]*>.*?"
                    r"<rdf:li[^>]*>(.*?)</rdf:li>",

                "description":
                    r"dc:description[^>]*>.*?"
                    r"<rdf:li[^>]*>(.*?)</rdf:li>",
            }

            for field, pattern in patterns.items():

                match = re.search(
                    pattern,
                    xmp_text,
                    re.DOTALL,
                )

                if match:
                    fields[field] = (
                        match.group(1)
                        .strip()
                    )

            return {
                "present": True,
                "fields": fields,
            }

        except Exception as e:

            return {
                "present": False,
                "fields": {},
                "error": str(e),
            }

    # ================================================================
    # C2PA
    # ================================================================

    @staticmethod
    def _analyze_c2pa(path: Path) -> dict:

        try:
            from c2pa import Reader

        except ImportError:

            return {
                "available": False,
                "present": None,
                "verified": None,
                "validation_results": None,
                "active_manifest": None,
                "manifests": [],
                "active_manifest_summary": None,
                "analysis_error": True,
                "error": (
                    "c2pa-python is not installed. "
                    "Run: pip install c2pa-python"
                ),
            }

        try:

            reader = Reader(str(path))

            manifest_json = reader.json()

            if not manifest_json:

                return {
                    "available": True,
                    "present": False,
                    "verified": False,
                    "validation_results": None,
                    "active_manifest": None,
                    "manifests": [],
                    "active_manifest_summary": None,
                    "analysis_error": False,
                    "error": None,
                }

            manifest_store = json.loads(
                manifest_json
            )

            # --------------------------------------------------------
            # Validation state
            # --------------------------------------------------------

            validation_state = None

            try:
                validation_state = str(
                    reader.get_validation_state()
                )

            except Exception:
                pass

            # --------------------------------------------------------
            # Validation results
            # --------------------------------------------------------

            validation_results = None

            try:

                validation_results = (
                    reader.get_validation_results()
                )

                validation_results = str(
                    validation_results
                )

            except Exception:
                pass

            # --------------------------------------------------------
            # Extract manifests
            # --------------------------------------------------------

            manifests = []

            raw_manifests = manifest_store.get(
                "manifests",
                {},
            )

            for manifest_id, manifest in (
                raw_manifests.items()
            ):

                summary = (
                    ProvenanceAnalyzer
                    ._extract_manifest_summary(
                        manifest_id,
                        manifest,
                    )
                )

                if summary:
                    manifests.append(summary)

            # --------------------------------------------------------
            # Active manifest
            # --------------------------------------------------------

            active_manifest_id = (
                manifest_store.get(
                    "active_manifest"
                )
            )

            active_manifest = None

            if active_manifest_id:
                active_manifest = (
                    raw_manifests.get(
                        active_manifest_id
                    )
                )

            active_summary = None

            if active_manifest:

                active_summary = (
                    ProvenanceAnalyzer
                    ._extract_manifest_summary(
                        active_manifest_id,
                        active_manifest,
                    )
                )

            return {
                "available": True,

                "present": True,

                "verified": validation_state,

                "validation_results": (
                    validation_results
                ),

                "active_manifest": (
                    active_manifest_id
                ),

                "manifests": manifests,

                "active_manifest_summary": (
                    active_summary
                ),

                "analysis_error": False,

                "error": None,
            }

        except Exception as e:

            # Important:
            # Parsing failure is NOT the same thing as
            # successfully determining that no C2PA exists.

            return {
                "available": True,

                "present": None,

                "verified": None,

                "validation_results": None,

                "active_manifest": None,

                "manifests": [],

                "active_manifest_summary": None,

                "analysis_error": True,

                "error": str(e),
            }

    # ================================================================
    # C2PA MANIFEST SUMMARY
    # ================================================================

    @staticmethod
    def _extract_manifest_summary(
        manifest_id,
        manifest,
    ):

        if not manifest:
            return None

        # ------------------------------------------------------------
        # Signature information
        # ------------------------------------------------------------

        signature_info = manifest.get(
            "signature_info",
            {},
        )

        # ------------------------------------------------------------
        # Basic manifest information
        # ------------------------------------------------------------

        claim_generator = manifest.get(
            "claim_generator"
        )

        title = manifest.get(
            "title"
        )

        format_name = manifest.get(
            "format"
        )

        # ------------------------------------------------------------
        # Actions
        # ------------------------------------------------------------

        actions = []

        assertions = manifest.get(
            "assertions",
            [],
        )

        for assertion in assertions:

            if not isinstance(assertion, dict):
                continue

            label = assertion.get("label")

            if label != "c2pa.actions":
                continue

            data = assertion.get(
                "data",
                {},
            )

            actions_data = data.get(
                "actions",
                [],
            )

            for action in actions_data:

                if not isinstance(action, dict):
                    continue

                action_info = {
                    "action": action.get(
                        "action"
                    ),

                    "when": action.get(
                        "when"
                    ),

                    "software_agent": action.get(
                        "softwareAgent"
                    ),

                    "parameters": action.get(
                        "parameters"
                    ),
                }

                action_info = {
                    key: value
                    for key, value
                    in action_info.items()
                    if value is not None
                }

                actions.append(
                    action_info
                )

        # ------------------------------------------------------------
        # Ingredients
        # ------------------------------------------------------------

        ingredients = []

        for ingredient in manifest.get(
            "ingredients",
            [],
        ):

            if not isinstance(
                ingredient,
                dict,
            ):
                continue

            ingredient_info = {
                "title": ingredient.get(
                    "title"
                ),

                "format": ingredient.get(
                    "format"
                ),

                "relationship": ingredient.get(
                    "relationship"
                ),

                "instance_id": ingredient.get(
                    "instance_id"
                ),
            }

            ingredient_info = {
                key: value
                for key, value
                in ingredient_info.items()
                if value is not None
            }

            ingredients.append(
                ingredient_info
            )

        # ------------------------------------------------------------
        # Return only observed information
        # ------------------------------------------------------------

        return {
            "manifest_id": manifest_id,

            "title": title,

            "format": format_name,

            "claim_generator": claim_generator,

            "signature": {
                "issuer": signature_info.get(
                    "issuer"
                ),

                "time": signature_info.get(
                    "time"
                ),

                "algorithm": signature_info.get(
                    "alg"
                ),
            },

            "actions": actions,

            "ingredients": ingredients,
        }