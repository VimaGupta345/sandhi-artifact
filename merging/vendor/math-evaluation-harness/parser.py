import re
import regex
import sympy
from typing import TypeVar, Iterable, List, Union, Any, Dict
from word2number import w2n
from utils import *

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def fix_sqrt(string):
    string = re.sub(r"\\sqrt(\d+)", r"\\sqrt{\1}", string)
    return string

def convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except:
        pass
    return text

unit_texts = ["east", "degree", "mph", "kmph", "ft", "m sqaure", "m", "east", "sq m", "deg", "mile", "q .", "monkey", "prime", "ratio", "profit of rs", "rd", "o", "gm", "p . m", "lb", "tile", "per", "dm", "lt", "gain", "ab", "way", "west", "a .", "b .", "c .", "d .", "e .", "f .", "g .", "h .", "t", "a", "h", "no change", "men", "soldier", "pie", "bc", "excess", "st", "inches", "noon", "percent", "by", "gal", "kmh", "c", "acre", "rise", "a . m", "th", "r 2", "sq", "mark", "l", "toy", "coin", "sq . m", "gallon", "f", "profit", "minw", "yr", "women", "feet", "am", "pm", "hr", "cu cm", "square", "v ", "are", "rupee", "rounds", "cubic", "cc", "mtr", "s", "ohm", "number", "kmph", "day", "hour", "minute", "min", "second", "man", "woman", "sec", "cube", "mt", "sq inch", "mp", "cm ", "hectare", "more", "sec", "unit", "cu . m", "cm 2", "rs .", "rs", "kg", "g", "month", "km", "m", "cm", "mm", "apple", "liter", "loss", "yard", "pure", "year", "increase", "decrease", "d", "less", "Surface", "litre", "pi sq m", "s .", "metre", "meter", "inch"]
unit_texts.extend([t + "s" for t in unit_texts])

def strip_string(string):
    string = str(string).strip()
    string = string.replace(",", "")  # linebreaks  
    string = string.rstrip(".")  # right . 
    string = string.replace("!", "")  # replace ! with 
    string = re.sub(r"\\array{.*?}", r"\\pmatrix{", string)
    string = re.sub(r"\\array", r"\\pmatrix", string)
    string = string.replace("\\bmatrix", "\\pmatrix")  # matrix
    string = string.replace("\\tfrac", "\\frac")
    string = string.replace("\\dfrac", "\\frac")  # replace \\tfrac and \\dfrac with \\frac
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\%", "%")
    string = string.replace("\\%", "%")  # remove $$ and %
    return string

def extract_multi_choice_answer(pred_str):
    if "Problem" in pred_str:
        pred_str = pred_str.split("Problem", 1)[0]

    pred_str = pred_str.replace("choice is", "answer is")
    patt = re.search(r"answer is \((.*)\)", pred_str.lower())
    if patt is not None:
        return patt.group(1)[0].upper()
    return ""

def extract_answer(pred_str, data_name):
    if data_name in ["mmlu_stem", "sat_math", "mathqa"]:
        return extract_multi_choice_answer(pred_str)

    if "final answer is" in pred_str and ". I hope" in pred_str:
        tmp = pred_str.split("final answer is ", 1)[1]
        pred = tmp.split(". I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split()[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    else:
        pattern = r"-?\d*\.?\d+"
        pred = re.findall(pattern, pred_str.replace(",", ""))
        if len(pred) >= 1:
            pred = pred[-1]
        else:
            pred = ""

    # Clean up the prediction
    pred = strip_string(pred)
    pred = convert_word_number(pred)

    # Handle special unit texts
    for text in unit_texts:
        if pred.endswith(" " + text):
            pred = pred[: -(len(text) + 1)]

    pred = pred.strip()
    return pred

def parse_ground_truth(sample, data_name):
    if data_name in ["mmlu_stem", "sat_math", "mathqa"]:
        gt_cot = sample["answer"]
        gt_ans = sample["answer"]
        return gt_cot, gt_ans

    # For math problems like GSM8K
    gt_cot = sample["answer"]
    ans_str = gt_cot

    if "####" in ans_str:
        gt_ans = ans_str.split("####")[-1].strip()
    else:
        gt_ans = extract_answer(gt_cot, data_name)

    return gt_cot, gt_ans

def parse_question(example, data_name):
    """
    Parse question from example data
    """
    if isinstance(example, dict):
        if 'question' in example:
            return example['question']
        elif 'problem' in example:
            return example['problem']
        elif 'input' in example:
            return example['input']

    # If example is a string, return as is
    return str(example)

# Math equality checking functions
def is_digit(s):
    try:
        float(s)
        return True
    except:
        return False

def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None

def math_equal(prediction, reference, include_percentage=True, is_close=True, timeout=True):
    if prediction is None or reference is None:
        return False

    if str(prediction) == str(reference):
        return True

    # Check if both are digits
    if is_digit(prediction) and is_digit(reference):
        prediction = parse_digits(prediction)
        reference = parse_digits(reference)
        if prediction is not None and reference is not None:
            if is_close:
                return abs(prediction - reference) < 1e-4
            else:
                return prediction == reference

    # Symbolic comparison
    try:
        if symbolic_equal(prediction, reference):
            return True
    except:
        pass

    return False

def symbolic_equal(a, b):
    def parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace(" ", ""))
            except:
                pass
        try:
            return f(s)
        except:
            pass
        return s

    a = parse(a)
    b = parse(b)

    try:
        if a.equals(b) or sympy.simplify(a-b) == 0:
            return True
    except:
        pass

    return False

def parse_latex(s):
    return latex2sympy(s)

def parse_expr(s):
    return sympy.sympify(s)

def latex2sympy(latex):
    return sympy.parsing.latex.parse_latex(latex)

# ===== ADDED FUNCTIONS FOR PAL SUPPORT =====

def extract_program(text):
    """
    Extract Python program from text generated by PAL prompting
    Handles multiple formats: ```python blocks, ``` blocks, and direct def solution() patterns
    """
    import re

    # Method 1: Extract code between ```python and ```
    if "```python" in text:
        pattern = r'```python\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()

    # Method 2: Extract code between ``` and ``` (without python tag)
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            for i in range(1, len(parts), 2):
                code = parts[i].strip()
                if "def solution" in code or "return" in code:
                    return code

    # Method 3: Look for def solution() pattern directly
    if "def solution()" in text:
        start = text.find("def solution()")
        remaining = text[start:]
        lines = remaining.split('\n')
        code_lines = []

        for line in lines:
            if line.strip() == "":
                code_lines.append(line)
                continue

            if line.startswith("def solution()"):
                code_lines.append(line)
                continue

            # Check if this line is part of the function
            if line.startswith("    ") or line.startswith("\t"):
                code_lines.append(line)
                if "return " in line.strip():
                    break
            else:
                # End of function
                break

        if len(code_lines) > 1:
            return '\n'.join(code_lines)

    # Method 4: Look for any Python-like code with proper indentation
    lines = text.split('\n')
    code_lines = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('def ') or stripped.startswith('import ') or stripped.startswith('from '):
            in_code_block = True
            code_lines.append(line)
        elif in_code_block:
            if stripped == '' or line.startswith('    ') or line.startswith('\t'):
                code_lines.append(line)
            elif stripped.startswith('return '):
                code_lines.append(line)
                break
            else:
                # End of code block
                break

    if code_lines:
        return '\n'.join(code_lines)

    # Fallback: return original text
    return text.strip()

def text_to_trajectory(text):
    """
    Convert text to trajectory format for compatibility with existing code
    """
    return [{"role": "program", "content": extract_program(text)}]

def run_execute(executor, result, prompt_type, data_name, execute=False):
    """
    FIXED VERSION: Execute the extracted program and return prediction and report
    """
    if not result or result == "error":
        return "", ""

    try:
        if prompt_type in ["cot"]:
            # For CoT method, extract answer from the reasoning text directly
            prediction = extract_answer(result, data_name)
            return prediction if prediction else "", ""
        elif prompt_type in ["pal"] and execute:
            # For PAL method, extract and execute the program
            program = extract_program(result)
            if program:
                try:
                    pred, report = executor.apply(program)
                    return pred if pred else "", report if report else ""
                except Exception as e:
                    return "", f"Execution error: {str(e)}"
            else:
                return "", "No program found"
        elif execute:
            # For other methods that need execution
            try:
                pred, report = executor.get_answer_from_stdout(result)
                return pred if pred else "", report if report else ""
            except Exception as e:
                return "", f"Execution error: {str(e)}"
        else:
            # Default case: extract answer from text
            prediction = extract_answer(result, data_name)
            return prediction if prediction else "", ""

    except Exception as e:
        return "", f"Error: {str(e)}"

# For backward compatibility, keep the old function name as well
def run_execute_fixed(executor, result, prompt_type, data_name, execute=False):
    """Alias for the fixed run_execute function"""
    return run_execute(executor, result, prompt_type, data_name, execute)
