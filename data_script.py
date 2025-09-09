import re
import boto3
from math import sqrt


def get_bbox(block):
    """Extract bounding box from Textract block dict."""
    return block["Geometry"]["BoundingBox"]


def center(bbox):
    """Return (x, y) center of a bounding box."""
    return (
        bbox["Left"] + bbox["Width"] / 2,
        bbox["Top"] + bbox["Height"] / 2
    )


def euclidean_distance(b1, b2):
    """Euclidean distance between two bounding boxes."""
    c1, c2 = center(b1), center(b2)
    return sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def find_nearest_block(response, regex, target_block_types=("SIGNATURE",)):
    """
    Works directly with Textract raw response (dicts).
    Finds the block nearest to regex-matching text.
    """
    pattern = re.compile(regex, re.IGNORECASE)
    blocks = response["Blocks"]
    results = []

    # Anchors: regex matches in LINE or WORD text
    anchors = [b for b in blocks if b["BlockType"] in ("LINE", "WORD") and pattern.search(b.get("Text", ""))]

    # Candidates: target block types
    candidates = [b for b in blocks if b["BlockType"] in target_block_types]

    for anchor in anchors:
        if not candidates:
            continue

        nearest = min(
            candidates,
            key=lambda c: euclidean_distance(get_bbox(anchor), get_bbox(c))
        )

        results.append({
            "anchor_text": anchor.get("Text", ""),
            "nearest_block_type": nearest["BlockType"],
            "nearest_box": get_bbox(nearest)
        })

    return results


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    client = boto3.client("textract")

    # Example: analyze document
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
