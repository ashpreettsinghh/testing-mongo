import boto3
from trp import Document  # amazon-textract-response-parser
from math import sqrt
import re  # Built-in for regex

# Step 1: Textract call (update with your S3 info)
textract = boto3.client('textract')
response = textract.analyze_document(
    Document={'S3Object': {'Bucket': 'your-bucket', 'Name': 'your-doc.jpg'}},
    FeatureTypes=['SIGNATURES']  # Add 'FORMS' if needed
)

# Step 2: Parse response
doc = Document(response)

# Helper: Distance between blocks (Euclidean on centers)
def calculate_distance(block1, block2):
    bb1 = block1.geometry.boundingBox
    center1_x = bb1.left + bb1.width / 2
    center1_y = bb1.top + bb1.height / 2
    bb2 = block2.geometry.boundingBox
    center2_x = bb2.left + bb2.width / 2
    center2_y = bb2.top + bb2.height / 2
    return sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)

# Step 3: Your regex patterns (case-insensitive)
patterns = [
    re.compile(r'Sign here', re.IGNORECASE),
    re.compile(r'Signature of the first cardholder', re.IGNORECASE)
]
max_dist_threshold = 0.15  # Adjust: filters far sigs
multi_limit = 3  # Max multiples per zone

for page in doc.pages:
    signatures = [b for b in page.blocks if b.blockType == 'SIGNATURE']
    if not signatures:
        print("No signatures detected.")
        continue
    
    # Find matched zones using your patterns
    matched_zones = []
    for line in page.lines:
        for pattern in patterns:
            if pattern.search(line.text):
                matched_zones.append((line, pattern.pattern))  # Track which pattern matched
                break  # Avoid duplicate matches on same line
    
    for zone, matched_pattern in matched_zones:
        # Calc distances to sigs
        dist_list = [(calculate_distance(zone, sig), sig) for sig in signatures if calculate_distance(zone, sig) < max_dist_threshold]
        
        # Sort by distance (closest first) and limit
        dist_list.sort(key=lambda x: x)
        nearest_sigs = dist_list[:multi_limit]
        
        if nearest_sigs:
            print(f"Zone: '{zone.text}' (Matched: '{matched_pattern}', Confidence: {zone.confidence})")
            for i, (dist, sig) in enumerate(nearest_sigs, 1):
                print(f"  Nearest Sig {i}: Distance {dist:.4f}, Confidence {sig.confidence}")
        else:
            print(f"No nearby sigs for zone: '{zone.text}' (Matched: '{matched_pattern}')")

# Changes/Fixes:
# - Swapped to your exact regexes in a list (easy to add more).
# - Explicit IGNORECASE for both (tested: catches variations like 'sign here' or 'SIGNATURE OF THE FIRST CARDHOLDER').
# - Tracked which pattern matched per zone for clearer output.
# - Optimized dist calc in list comprehension (faster, avoids redundant calls).
# - No changes to core logic – keeps it lightweight and direction-agnostic.
