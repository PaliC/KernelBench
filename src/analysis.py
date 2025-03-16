################################################################################
# Helpers for Analysis
################################################################################
import numpy as np

from functools import cache
from transformers import AutoTokenizer
import utils
import re


def pass_at_k(n, c, k):
    """
    A numerically stable script for calculating an unbiased estimate of pass@k
    Referenced from HumanEval: https://arxiv.org/abs/2107.03374
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def get_token_count(text: str, tokenizer: AutoTokenizer) -> int:
    assert isinstance(text, str), "can only tokenize strings but got {}".format(type(text))
    return len(tokenizer.encode(text))


def extract_all_cuda_sources(file_content: str) -> list[str]:
    """
    Extract all CUDA sources wrapped in triple quotes.
    
    Returns:
        list[str]: List of all extracted CUDA source code blocks
    """
    pattern = r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*"""(.*?)"""'
    matches = re.findall(pattern, file_content, re.DOTALL)
    return [match.strip() for match in matches]


def get_cuda_tokens(kernel_src: str, tokenizer: AutoTokenizer) -> int:
    """
    Count number of all CUDA tokens in the kernel
    """
    all_cuda_code = extract_all_cuda_sources(kernel_src)
    num_cuda_tokens = sum(get_token_count(code, tokenizer) for code in all_cuda_code)
    return num_cuda_tokens


################################################################################
# Error Analysis
# Migrated from arxiv paper analysis 
################################################################################


################################################################################
# CUDA
################################################################################
def get_compilation_error_type(eval_result_metadata: dict) -> str | None:
    """
    Assume eval_result Compilation is False

    Assume eval_result_metadata is a dict, not wrapped in `eval_0`
    just the straight up metadata of EvaluationResult
    Coarse grain compilation error type

    There are 3 (high level) types of compilation errors:
    - compilation_error: nvcc not able to compile, likely syntax or import error
    - cuda_error: CUDA access error
    - other_error: other error

    There shouldn't be a runtime_error (that should be correctness issue)
    """

    error_type = None
    # List of possible error types to check
    error_types = ['compilation_error', 'other_error', 'cuda_error']

    for error_type in error_types:
        if error_type in eval_result_metadata:
            error_type = error_type
            break

    if error_type is None:
        raise Exception("No known error type found in eval result:", eval_result_metadata)

    return error_type

def get_correctness_issue_type(eval_result_metadata: dict) -> str | None:
    """
    Coarse-grained correctness issue type
    Assume eval_result Correctness is False

    There are 3 (high level) types of correctness issues:
    - correctness_issue: correctness issue
        -> output_value_mismatch: value does not match expected output
        -> output_shape_mismatch: output shape does not match expected output shape
    - runtime_error: runtime error
    """

    correctness_issue_type = None
    if 'correctness_issue' in eval_result_metadata:
        specific_correctness_issue = eval_result_metadata['correctness_issue']

        if 'output mismatch' in specific_correctness_issue.lower():
            correctness_issue_type = 'output_value_mismatch'
        elif 'output shape mismatch' in specific_correctness_issue.lower():
            correctness_issue_type = 'output_shape_mismatch'
        else:
            raise Exception("Unknown correctness issue type:", specific_correctness_issue)
    elif 'runtime_error' in eval_result_metadata:
        correctness_issue_type = 'runtime_error'    
    else:
        raise Exception("No known correctness issue type found in eval result:", eval_result_metadata)
    
    return correctness_issue_type


################################################################################
# Triton
################################################################################

def get_correctness_issue_type_triton(eval_result_metadata: dict) -> str | None:
    """
    Coarse + Fine-grained correctness issue type
    Assume eval_result Correctness is False

    Careful: Triton is jit compiled so a lot of the runtime errors contain
    all kinds of errors that needs more detailed analysis

    There are 3 (high level) types of correctness issues:
    - correctness_issue: correctness issue
        -> output_value_mismatch: value does not match expected output
        -> output_shape_mismatch: output shape does not match expected output shape
    - runtime_error: runtime error
        -> TBD
    """
    def keywords_in_error(error_str: str, keywords: list[str]) -> bool:
        return any(keyword in error_str.lower() for keyword in keywords)
    
    correctness_issue_type = None
    if 'correctness_issue' in eval_result_metadata:
        specific_correctness_issue = eval_result_metadata['correctness_issue']
        if 'output mismatch' in specific_correctness_issue.lower():
            correctness_issue_type = 'output_value_mismatch'
        elif 'output shape mismatch' in specific_correctness_issue.lower():
            correctness_issue_type = 'output_shape_mismatch'
        else:
            raise Exception("Unknown correctness issue type:", specific_correctness_issue)
    elif 'runtime_error' in eval_result_metadata:
        # Triton Runtime Error s
        # Careful they might still be a correctness issue
        
        specific_runtime_error = eval_result_metadata['runtime_error']
        if keywords_in_error(specific_runtime_error, ['not defined']):
            correctness_issue_type = 'undefined_variable'
        elif keywords_in_error(specific_runtime_error, ['expected size', 'number of dimensions', 'shapes']):
            correctness_issue_type = 'dimension_issues'
        elif keywords_in_error(specific_runtime_error, ['cpu']):
            correctness_issue_type = 'device_error'
        else:
            print(f"Unknown runtime error: {specific_runtime_error}")
            # raise Exception("Unknown runtime error:", specific_runtime_error)
            correctness_issue_type = 'runtime_error'
    else:
        raise Exception("No known correctness issue type found in eval result:", eval_result_metadata)
    
    return correctness_issue_type




# def get_error_type_from_evaluation_result(
#     eval_result: dict, 
#     key: None | str = None 
#     # in case meta data is not a flat json dict but wrapped such as `eval_0`
# ) -> str | None:
#     """
#     For CUDA
#     Created from an json object
#     """
#     # If it compiles and is correct, no error
#     if eval_result.compiles and eval_result.correct:
#         return None

#     # parse the eval result metaa
#     eval_result_metadata = eval_result.metadata

#     # eval_result_metadata might not be a flat json dict
#     if key is not None:
#         eval_result_metadata = eval_result_metadata[key]

#     if not eval_result.compiles:
#         return get_compilation_error_type(eval_result_metadata)

#     if not eval_result.correct:
#         return get_correctness_issue_type(eval_result_metadata)

#     raise Exception("No known error type found in eval result:", eval_result.metadata)




# def get_error_type_from_evaluation_result_triton(
#     eval_result: dict, 
#     key: None | str = None 
#     # in case meta data is not a flat json dict but wrapped such as `eval_0`
# ) -> str | None:
#     """
#     Created from an json object
#     """
#     # If it compiles and is correct, no error
#     if eval_result.get("compiled", False) and eval_result.get("correct", False):
#         return None

#     # parse the eval result metaa
#     eval_result_metadata = eval_result.get("metadata", {})

#     # eval_result_metadata might not be a flat json dict
#     if key is not None:
#         eval_result_metadata = eval_result_metadata[key]

#     if not eval_result.get("compiled", False):
#         return get_compilation_error_type(eval_result_metadata)

#     if not eval_result.get("correct", False):
#         return get_correctness_issue_type(eval_result_metadata)

#     raise Exception("No known error type found in eval result:", eval_result.metadata)