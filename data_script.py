from datamodel_code_generator import InputFileType, generate

def generate_pydantic_from_json(json_file_path: str, output_file_path: str):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = f.read()

    code = generate(
        input_text=json_data,
        input_file_type=InputFileType.Json,
        # You can add options here like target Python version, class name, etc.
    )

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f"Pydantic model code generated and saved to {output_file_path}")

# Example usage
generate_pydantic_from_json('your_file.json', 'model.py')
