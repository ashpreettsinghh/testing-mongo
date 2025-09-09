import re
from math import sqrt

def center(box):
    return (box.left + box.width/2, box.top + box.height/2)

def euclidean_distance(b1, b2):
    c1, c2 = center(b1), center(b2)
    return sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

def find_nearest_block(doc, regex, target_block_types=("SIGNATURE",)):
    """
    Find the nearest target block (SIGNATURE, WORD, etc.) to a regex-matched text block.
    Checks in all directions (no top/bottom/left/right restriction).
    """
    results = []
    pattern = re.compile(regex, re.IGNORECASE)

    for page in doc.pages:
        # anchors = text blocks matching regex
        anchors = [line for line in page.lines if pattern.search(line.text)]
        
        for anchor in anchors:
            candidates = [b for b in page.content if b.block_type in target_block_types]
            if not candidates:
                continue

            # find nearest by Euclidean distance
            nearest = min(
                candidates, 
                key=lambda c: euclidean_distance(anchor.geometry.boundingBox, c.geometry.boundingBox)
            )

            results.append({
                "anchor_text": anchor.text,
                "nearest_block_type": nearest.block_type,
                "nearest_box": nearest.geometry.boundingBox
            })

    return results

matches = find_nearest_block(doc, r"(Authorized Signatory|Approver)", target_block_types=("SIGNATURE",))

for m in matches:
    print("Anchor:", m["anchor_text"])
    print("Nearest block type:", m["nearest_block_type"])
    print("Bounding box:", m["nearest_box"])
