import re
import boto3
from math import sqrt
from trp import Document


# --------------------------
# Helpers
# --------------------------
def get_bbox(block):
    """
    Normalize bounding box access for TRP objects and raw Textract dicts.
    """
    # TRP object
    if hasattr(block, "geometry") and hasattr(block.geometry, "boundingBox"):
        return block.geometry.boundingBox
    # Raw dict
    elif isinstance(block, dict) and "Geometry" in block:
        return block["Geometry"]["BoundingBox"]
    else:
        raise TypeError(f"Unsupported block type: {type(block)}")


def center(bbox):
    """Return (x,y) center of a bounding box."""
    if isinstance(bbox, dict):
        return (bbox["Left"] + bbox["Width"] / 2, bbox["Top"] + bbox["Height"] / 2)
    else:  # TRP boundingBox object
        return (bbox.left + bbox.width / 2, bbox.top + bbox.height / 2)


def euclidean_distance(b1, b2):
    """Euclidean distance between two bounding boxes."""
    c1, c2 = center(b1), center(b2)
    return sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


# --------------------------
# Main function
# --------------------------
def find_nearest_block(response, regex, target_block_types=("SIGNATURE",)):
    """
    Given a Textract response and regex, find the nearest block(s) of target_block_types.
    Works for both TRP objects and raw Textract JSON.
    """
    results = []
    pattern = re.compile(regex, re.IGNORECASE)

    # Wrap with TRP for convenience
    doc = Document(response)

    for page in doc.pages:
        # Anchors: regex matches in text lines
        anchors = [line for line in page.lines if pattern.search(line.text)]

        # Candidates: matching block types
        candidates = [b for b in page.blocks if b.block_type in target_block_types]

        for anchor in anchors:
            if not candidates:
                continue

            # Pick nearest
            nearest = min(
                candidates,
                key=lambda c: euclidean_distance(get_bbox(anchor), get_bbox(c))
            )

            results.append({
                "anchor_text": anchor.text,
                "nearest_block_type": nearest.block_type,
                "nearest_box": get_bbox(nearest)
            })

    return results


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    client = boto3.client("textract")

    # Example: analyze a form
    response = client.analyze_document(
        Document={"S3Object": {"Bucket": "your-bucket", "Name": "your-doc.pdf"}},
        FeatureTypes=["FORMS", "SIGNATURES"]
    )

    matches = find_nearest_block(
        response,
        r"(Authorized Signatory|Approver)",
        target_block_types=("SIGNATURE",)
    )

    for m in matches:
        print("Anchor:", m["anchor_text"])
        print("Nearest block type:", m["nearest_block_type"])
        print("Bounding box:", m["nearest_box"])
        print("----")
