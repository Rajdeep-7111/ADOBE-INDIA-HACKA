import os
import json
from extractor import process_pdf

# Define input and output directory paths
INPUT_DIR = "input"
OUTPUT_DIR = "output"

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Iterate over all files in input directory
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            input_path = os.path.join(INPUT_DIR, filename)
            output_filename = filename.rsplit(".", 1)[0] + ".json"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            try:
                # Extract title and outline from PDF
                result = process_pdf(input_path)

                # Save result to JSON
                with open(output_path, "w", encoding="utf-8") as out_file:
                    json.dump(result, out_file, indent=2, ensure_ascii=False)

                print(f"[✓] Processed: {filename}")
            except Exception as e:
                print(f"[✗] Failed to process {filename}: {e}")

if __name__ == "__main__":
    main()