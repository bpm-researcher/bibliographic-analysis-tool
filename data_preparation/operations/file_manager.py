import os

OUTPUT_DIR = "biliographic_analysis_on_indicators/data_preparation/outputs"
OUTPUT_PREFIX = "output_"

def get_next_output_filename(format):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    existing = [file for file in os.listdir(OUTPUT_DIR) if file.startswith(OUTPUT_PREFIX)]
    existing_nums = []

    for file in existing:
        try:
            num = (file.replace(OUTPUT_PREFIX, ""))
            num = int(num.replace(format, ""))
            existing_nums.append(num)
        except ValueError:
            continue

    next_num = max(existing_nums, default=0) + 1
    return os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}{next_num}{format}")