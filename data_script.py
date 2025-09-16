form_fields = []
blocks_map = {block.id: block for page in doc.pages for block in page.form.fields}

def find_parent_key(block, blocks_map):
    # Iterate all blocks to find one that has the current block's id in CHILD relationships
    for blk in blocks_map.values():
        if blk.block_type == "KEY_VALUE_SET" and "KEY" in blk.entity_types:
            for rel in blk.relationships or []:
                if rel.type == "CHILD" and block.id in rel.ids:
                    # Found a parent key block, get its text children
                    return blk.text
    return None

for page in doc.pages:
    for field in page.form.fields:
        if field.key and field.value:
            key = field.key.text.strip() if field.key.text else None
            value = field.value.text.strip() if field.value.text else None

            parent_key = find_parent_key(field.key, blocks_map) if field.key else None
            parent_value = None  # You can similarly attempt to find parent value if needed

            field_data = {
                "parentKey": parent_key,
                "parentValue": parent_value,
                "keyId": field.key.id,
                "key": key,
                "value": value,
                "bbox": field.key.geometry.bounding_box._asdict() if field.key.geometry else None
            }
            form_fields.append(field_data)
