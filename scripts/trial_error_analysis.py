import json, os
import pydra
from pydra import REQUIRED, Config
from src.dataset import construct_kernelbench_dataset
from src.analysis import get_compilation_error_type, get_correctness_issue_type_triton
from collections import Counter
from tabulate import tabulate

"""
Benchmark Eval Analysis

This script shows how to conduct analysis for model performance on KernelBench

Given generations and eval results, this script will compute the following:
- Success rate (compiled and correctness)
- Error analysis for failed compilations and incorrect results

Usage:
```
python3 scripts/trial_error_analysis.py run_dir=<run_dir> run_name=<run_name> level=<level> hardware=<hardware> baseline=<baseline>
```
hardware + baseline should correspond to the results/timing/hardware/baseline.json file   

""" 

class AnalysisConfig(Config):
    def __init__(self):
        self.run_dir = "runs"
        self.run_name = REQUIRED # name of the run to evaluate
        self.level = REQUIRED # level to evaluate

        # self.hardware = REQUIRED # hardware to evaluate
        # self.baseline = REQUIRED # baseline to compare against

    def __repr__(self):
        return f"AnalysisConfig({self.to_dict()})"

def patch(eval_results, dataset):
    """
    Patch the eval results with the dataset
    """
    for pid in range(1, len(dataset) + 1):
        if str(pid) not in eval_results:
            eval_results[str(pid)] = {
                "sample_id": 0, 
                "compiled": False, 
                "correctness": False, 
                "metadata": {},
                "runtime": -1.0, 
                "runtime_stats": {}
            }
    return eval_results

def analyze_error_types(run_dir, run_name, level):
    """
    Analyze the error types for a run of a particular level
    """

    dataset = construct_kernelbench_dataset(level)

    # load json
    eval_file_path = f'{run_dir}/{run_name}/eval_results.json'
    assert os.path.exists(eval_file_path), f"Eval file does not exist at {eval_file_path}"

    # baseline_file_path = f'results/timing/{hardware}/{baseline}.json'
    # assert os.path.exists(baseline_file_path), f"Baseline file does not exist at {baseline_file_path}"

    with open(eval_file_path, 'r') as f:
        eval_results = json.load(f)

    # with open(baseline_file_path, 'r') as f:
    #     baseline_results = json.load(f)

    # Initialize counters
    total_count = len(dataset)
    total_eval = len(eval_results)
    compiled_count = 0
    correct_count = 0

    # Patch the eval results
    eval_results = patch(eval_results, dataset)

    # Initialize error counters
    compilation_error_types = Counter()
    correctness_issue_types = Counter()

    # Count results and errors
    for pid, entry in eval_results.items():

        if entry["compiled"] == True:
            compiled_count += 1
        if entry["correctness"] == True:
            correct_count += 1
        
        # Analyze errors
        if not entry["compiled"]:
            try:
                error_type = get_compilation_error_type(entry["metadata"])
                compilation_error_types[error_type] += 1
            except Exception as e:
                compilation_error_types["unknown"] += 1
        elif not entry["correctness"]:
            try:
                # print(f"Entry: {entry}")
                error_type = get_correctness_issue_type_triton(entry["metadata"])
                correctness_issue_types[error_type] += 1
            except Exception as e:
                correctness_issue_types["unknown"] += 1

    # Print results
    print("-" * 128)
    print(f"Eval Summary for {run_name}")
    print("-" * 128)
    print(f"Total test cases with Eval Results: {total_eval} out of {total_count}")
    print(f"Successfully compiled: {compiled_count}")
    print(f"Functionally correct: {correct_count}")

    print(f"\nSuccess rates:")
    print(f"Compilation rate: {compiled_count/total_count*100:.1f}%")
    print(f"Correctness rate: {correct_count/total_count*100:.1f}%") 

    # Print error analysis
    print("\nError Analysis:")
    print("Compilation Error Types:")
    if total_count - compiled_count > 0:
        compilation_error_table = [[error_type, count, f"{count/(total_count-compiled_count)*100:.1f}%"] 
                                for error_type, count in compilation_error_types.most_common()]
        print(tabulate(compilation_error_table, headers=["Error Type", "Count", "% of Failed Compilations"], tablefmt="grid"))
    else:
        print("No compilation errors found.")
    
    print("\nCorrectness Issue Types:")
    if compiled_count - correct_count > 0:
        correctness_issue_table = [[issue_type, count, f"{count/(compiled_count-correct_count)*100:.1f}%"] 
                                for issue_type, count in correctness_issue_types.most_common()]
        print(tabulate(correctness_issue_table, headers=["Issue Type", "Count", "% of Incorrect Results"], tablefmt="grid"))
    else:
        print("No correctness issues found.")


@pydra.main(base=AnalysisConfig)
def main(config: AnalysisConfig):
    analyze_error_types(config.run_dir, config.run_name, config.level)

if __name__ == "__main__":
    main()