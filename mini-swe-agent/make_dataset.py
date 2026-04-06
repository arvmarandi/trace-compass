import re
import json
from pathlib import Path
from collections import defaultdict 

def parse_call_stack(trace_text):
    """Extract file, line number, and function name from stack trace"""
    pattern = r'File "(.*?)", line (\d+), in (.*)'
    call_stack = []
    
    for match in re.finditer(pattern, trace_text):
        file_path = match.group(1)
        line_num = match.group(2)
        func_name = match.group(3)
        
        if "/testbed/" in file_path and "/miniconda" not in file_path:
            call_stack.append({
                "file": file_path,
                "line": int(line_num),
                "function": func_name
            })
            
    return call_stack

def is_valid_bug_trace(parsed_stack, raw_error_msg):
    """Determine if the extracted stack is a 'real bug'"""
    if not parsed_stack:
        return False
        
    last_call = parsed_stack[-1]
    crash_file = last_call["file"]
    
    if "/testbed/" not in crash_file: 
        return False
        
    # noise_errors = ["ImportError", "ModuleNotFoundError", "SyntaxError", "NameError"]
    # if any(noise in raw_error_msg for noise in noise_errors):
    #     return False
        
    return True

def process_all_trajectories(base_dir):
    # Instead of an empty list, create a dictionary with instance_id as the key
    grouped_data = defaultdict(list)
    
    base_path = Path(base_dir)
    json_files = list(base_path.rglob("*.traj.json"))
    print(f"Total {len(json_files)} traj.json files found.\n")
    
    for filepath in json_files:
        print(f"Parsing: {filepath.name} ...")
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
                
        # Get the problem ID of the current file
        instance_id = data.get("instance_id", "unknown")
        
        for msg in data.get("messages", []):
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                
                if "Traceback (most recent call last):" in content or "FAILED" in content:
                    match = re.search(r'<output>\n?(.*?)\n?</output>', content, re.DOTALL)
                    if match:
                        raw_output = match.group(1).strip()
                        parsed_stack = parse_call_stack(raw_output)
                        
                        if is_valid_bug_trace(parsed_stack, raw_output):
                            # Push the data into the corresponding instance_id room
                            grouped_data[instance_id].append({
                                "call_stack": parsed_stack,
                                "raw_error": raw_output
                            })

    output_file = "sbest_grouped_dataset.json"
    
    # Convert defaultdict to a regular dictionary and save as JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dict(grouped_data), f, indent=2, ensure_ascii=False)
    
    # Calculate how many 'problems (Instances)' have been organized, and how many 'error logs' there are
    total_traces = sum(len(traces) for traces in grouped_data.values())
    print(f"\n COMPLETED: Organized {len(grouped_data)} problems (Instances), and extracted {total_traces} stack traces and saved to {output_file}.")  

# Execute EXAMPLE
process_all_trajectories("./outputs")