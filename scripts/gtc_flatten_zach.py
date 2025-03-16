import json
import os
from pathlib import Path

def process_jsonl_file(json_file_path,
                      output_dir: str = "generations",
                      level: str = "level1",
                      enable_write: bool = True,
                      ):
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    count = 0
    # Process each line in the JSONL file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            if 'source_file' in entry:
                # Get the last part of the source_file path as filename
                source_path = Path(entry['source_file'])
                source_file_name = source_path.name
                problem_num = source_file_name.split('_')[0]  # Extract problem number
                filename = f"level_1_problem_{problem_num}_sample_0_kernel.py"
                output_path = output_dir / filename

                print(f"Found for {source_file_name} Writing generation to: {output_path}")                

            # Write the generation content to a file
            if 'generation' in entry and enable_write:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(entry['generation'])
                print(f"Written generation to: {output_path}")
                count += 1

    print(f"Total generations written: {count}")

if __name__ == "__main__":
    # Specify your JSON file path here
    level = "level1"
    run_dir = "gtc_analysis/run_zach_1"
    json_file_path = os.path.join(run_dir, "zach_src.jsonl")
    output_dir = run_dir    
    process_jsonl_file(json_file_path, output_dir, level=level)

