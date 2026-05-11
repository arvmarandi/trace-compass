#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import sys
import concurrent.futures
import json
import random
import re
import threading
import time
import traceback
from pathlib import Path
import difflib

import typer
from jinja2 import StrictUndefined, Template
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c swebench.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c swebench.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c swebench.yaml -c agent.max_iterations=50[/bold green]
"""

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
    "rebench": "nebius/SWE-rebench",
}

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()


class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        return super().step()


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None) or instance.get("docker_image", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def get_sb_environment(config: dict, instance: dict) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)
    if env_config["environment_class"] in ["docker", "swerex_modal"]:
        env_config["image"] = image_name
    elif env_config["environment_class"] in ["singularity", "contree"]:
        env_config["image"] = "docker://" + image_name

    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting environment")

    agent = None
    exit_status = None
    result = None
    extra_info = {}

    try:
        env = get_sb_environment(config, instance)
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **config.get("agent", {}),
        )

        # PART 1: Focal Function Localization
        progress_manager.update_instance_status(instance_id, "Localizing program functions")
        prog_files = find_program_files(env, exclude_test_files=True)
        llm_prog_files = []
        for _ in range(5):
            raw = agent.run_func_loc1(task, prog_files)
            raw = raw.replace('\\\\n', '\n').replace('\\n', '\n').strip().rstrip('\\').strip("'\"")
            llm_prog_files = [f.strip(' "\'').strip() for f in raw.split('\n') if f.strip()]
            if (len(prog_files) >= 10 and len(llm_prog_files) == 10) or len(prog_files) < 10:
                break

        validated_prog_files = [
            p for f in llm_prog_files
            if (p := find_closest_paths(f, prog_files)) is not None
        ]
        prog_functions = find_functions(env, validated_prog_files)
        prog_function_paths = agent.run_func_loc2(task, prog_functions)

        # PART 2: Test Function Localization
        progress_manager.update_instance_status(instance_id, "Localizing test functions")
        test_files = find_test_files(env)
        llm_test_files = []
        for _ in range(5):
            raw = agent.run_test_func_loc1(task, test_files)
            raw = raw.replace('\\\\n', '\n').replace('\\n', '\n').strip().rstrip('\\').strip("'\"")
            llm_test_files = [f.strip(' "\'').strip() for f in raw.split('\n') if f.strip()]
            if (len(test_files) >= 10 and len(llm_test_files) == 10) or len(test_files) < 10:
                break

        validated_test_files = [
            p for f in llm_test_files
            if (p := find_closest_paths(f, test_files)) is not None
        ]
        test_functions = find_test_functions(env, validated_test_files)
        test_function_paths = agent.run_test_func_loc2(task, test_functions)

        # PART 3: Test Generation
        progress_manager.update_instance_status(instance_id, "Generating reproduction tests")
        prog_bodies = get_function_bodies(env, prog_function_paths)
        test_bodies = get_function_bodies(env, test_function_paths)
        info = agent.run(task, prog_bodies + "\n\n" + test_bodies)
        exit_status = info.get("exit_status")
        result = info.get("submission")

        # PART 4: Generate Stack Traces
        if result:
            progress_manager.update_instance_status(instance_id, "Capturing stack traces")
            try:
                apply_patch_to_env(env, result)
                patch_test_files = extract_test_files_from_patch(result)
                if patch_test_files:
                    repo_id = instance.get("repo", "")
                    version = instance.get("version", "")
                    raw_output = run_tests_in_env(env, patch_test_files, repo_id, version)
                    extra_info["stack_traces"] = extract_stack_traces(raw_output)
                    extra_info["test_output"] = raw_output
                    settrace = run_settrace_on_failing_tests(env, raw_output, repo_id, patch_test_files)
                    if settrace:
                        extra_info["settrace_traces"] = settrace
            except Exception as trace_err:
                logger.warning(f"Stack trace generation failed for {instance_id}: {trace_err}")

    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, ""
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        traj_path = instance_dir / f"{instance_id}.traj.json"
        instance_dir.mkdir(parents=True, exist_ok=True)
        if agent is not None:
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": result,
                        **extra_info,
                    },
                    "instance_id": instance_id,
                },
            )
        else:
            traj_path.write_text(json.dumps({
                "info": {"exit_status": exit_status, "submission": result, **extra_info},
                "instance_id": instance_id,
                "messages": [],
            }, indent=2))
        logger.info(f"Saved trajectory to '{traj_path}'")
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False, limit: int = 0
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    if limit > 0 and len(instances) > limit:
        instances = instances[:limit]
        logger.info(f"Instance limit: {limit} (final {len(instances)} instances)")
    return instances


def select_instances(instances: list[dict], *, instance_ids: str = "") -> list[dict]:
    """Select a subset of instances by comma-separated IDs and/or indices."""
    if instance_ids:
        ids = {item.strip() for item in instance_ids.split(",") if item.strip()}
        instances = [instance for instance in instances if instance["instance_id"] in ids]
        logger.info(f"Instance IDs selector reduced to {len(instances)} instances")
    return instances

# return all the program files in the repository
def find_program_files(env: Environment, exclude_test_files: bool = False) -> list[str]:
    program_files = set()
    test_files = set()
    
    # If excluding test files, first find all test files
    if exclude_test_files:
        test_files = set(find_test_files(env))
    
    # Find all Python files in the repository
    try:
        out = env.execute({"command": "find /testbed -type f -name '*.py' 2>/dev/null | head -200"})
        if out["returncode"] == 0:
            all_files = out["output"].strip().split("\n")
            program_files.update(all_files)
    except Exception:
        pass
    
    # clean up and filter results
    program_files = {f for f in program_files if f.strip() and f.startswith("/testbed/")}
    program_files = {f.replace("/testbed/", "") for f in program_files if f}
    
    # exclude test files if requested
    if exclude_test_files:
        program_files = program_files - test_files
    
    # Remove empty strings and sort
    result = sorted(list(program_files))
    return result


def find_test_files(env: Environment) -> list[str]:
    test_files = set()

    # find files in test directories
    try:
        out = env.execute({"command": "find /testbed -type f -path '*/test*' -name '*.py' 2>/dev/null | head -50"})
        if out["returncode"] == 0:
            test_files.update(out["output"].strip().split("\n"))
    except Exception:
        pass

    # find files with 'test' in filename
    try:
        out = env.execute({"command": "find /testbed -type f -name '*test*.py' 2>/dev/null | head -50"})
        if out["returncode"] == 0:
            test_files.update(out["output"].strip().split("\n"))
    except Exception:
        pass

    # find Python files containing test patterns
    try:
        out = env.execute(
            {
                "command": "find /testbed -type f -name '*.py' -exec grep -E -l 'def test_|class Test' {} \\; 2>/dev/null | head -50"
            }
        )
        if out["returncode"] == 0:
            test_files.update(out["output"].strip().split("\n"))
    except Exception:
        pass

    # clean the paths
    test_files = {f for f in test_files if f.strip() and f.startswith("/testbed/")}
    test_files = {f.replace("/testbed/", "") for f in test_files if f}

    result = sorted(list(test_files))
    return result


def find_test_functions(env: Environment, test_files: list[str]) -> list[str]:
    test_functions = []
    for file_path in test_files:
        if not file_path.strip():
            continue
        
        file_path = file_path.lstrip('/')
        full_path = f"/testbed/{file_path}"
        
        try:
            out = env.execute({"command": f"grep -n 'def test_' '{full_path}' 2>/dev/null || true"})
            if out["returncode"] == 0 and out["output"].strip():
                current_class = None
                # get class context too
                full_out = env.execute({"command": f"grep -n 'class \|def test_' '{full_path}' 2>/dev/null || true"})
                for line in full_out["output"].strip().split("\n"):
                    content = line.split(":", 1)[-1] if ":" in line else line
                    if "class " in content and not content.strip().startswith("def"):
                        current_class = content.strip().split("class ")[1].split("(")[0].strip()
                    elif "def test_" in content:
                        stripped = content.strip()
                        if not stripped.startswith("def test_") or "(" not in stripped:
                            continue
                        func_name = stripped[4:stripped.index("(")].strip()
                        if current_class:
                            test_functions.append(f"{file_path}::{current_class}::{func_name}")
                        else:
                            test_functions.append(f"{file_path}::{func_name}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return sorted(test_functions)


def find_functions(env: Environment, file_paths: list[str]) -> list[str]:
    functions = []
    for file_path in file_paths:
        if not file_path.strip():
            continue
        file_path = file_path.lstrip('/')
        full_path = f"/testbed/{file_path}"

        script = f"""import ast
with open('{full_path}') as f:
    source = f.read()
tree = ast.parse(source)
file_path = '{file_path}'
for node in tree.body:
    if isinstance(node, ast.ClassDef):
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                print(f"{{file_path}}::{{node.name}}::{{item.name}}")
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        print(f"{{file_path}}::{{node.name}}")
"""
        try:
            # Write script to temp file and execute
            env.execute({"command": f"cat > /tmp/extract_funcs.py << 'PYEOF'\n{script}\nPYEOF"})
            out = env.execute({"command": "python3 /tmp/extract_funcs.py"})
            if out.get("output", "").strip():
                for line in out["output"].strip().split("\n"):
                    if line.strip():
                        functions.append(line.strip())
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    return sorted(functions)

def find_closest_paths(llm_guess, all_repo_files):
    # 1. pre-process: LLMs often use dots (astropy.modeling.core) instead of slashes
    clean_guess = llm_guess.replace('.', '/').strip()
    if not clean_guess.endswith('.py') and '/' in clean_guess:
        clean_guess += '.py'

    # 2. happy path
    if clean_guess in all_repo_files:
        return clean_guess

    # 3. Levenshtein distance
    # cutoff=0.6 is a good balance for "close enough"
    matches = difflib.get_close_matches(clean_guess, all_repo_files, n=1, cutoff=0.6)
    
    if matches:
        return matches[0]

    # 4. if the path is wrong, but the filename is unique in the repo
    llm_filename = clean_guess.split('/')[-1]
    filename_matches = [f for f in all_repo_files if f.endswith(llm_filename)]
    
    if len(filename_matches) == 1:
        return filename_matches[0]

    return None # Truly not found

def get_function_body(env, signature: str) -> str:
    parts = signature.split("::")
    if len(parts) == 2:
        file_path, func_name = parts
    elif len(parts) == 3:
        file_path, _, func_name = parts
    else:
        return ""

    full_path = f"/testbed/{file_path}"

    # find the line number of the function definition
    find_obs = env.execute({
        "command": f"grep -n 'def {func_name}' '{full_path}' 2>/dev/null | head -1"
    })
    line_str = find_obs.get("output", "").split(":")[0].strip()
    if not line_str.isdigit():
        return ""

    start_line = int(line_str)

    # extract the function body using the AST
    extract_cmd = f"""python3 - <<'EOF'
import ast, sys

with open('{full_path}') as f:
    source = f.read()

tree = ast.parse(source)
lines = source.splitlines()

for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.name == '{func_name}' and node.lineno == {start_line}:
            print('\\n'.join(lines[node.lineno - 1:node.end_lineno]))
            sys.exit(0)
EOF"""

    body_obs = env.execute({"command": extract_cmd})
    return body_obs.get("output", "").strip()


def get_function_bodies(env, signatures_raw: str) -> str:
    normalized = signatures_raw.replace('\\n', '\n').replace('\\\\n', '\n') # LLM hallucinates backslashes for newline characters
    signatures = [s.strip() for s in normalized.split("\n") if "::" in s]
    if not signatures:
        return ""

    results = []
    for sig in signatures:
        body = get_function_body(env, sig)
        if body:
            results.append(f"--- {sig} ---\n{body}")

    return "\n\n".join(results)

def apply_patch_to_env(env: Environment, patch_str: str) -> None:
    """Write patch to container and apply it with git apply."""
    env.execute({"command": "cat > /tmp/model.patch << 'PATCHEOF'\n" + patch_str + "\nPATCHEOF"})
    out = env.execute({"command": "cd /testbed && git apply /tmp/model.patch 2>&1"})
    logger.debug(f"git apply output: {out.get('output', '')}")
    if out.get("returncode", 0) != 0:
        # git apply refuses to create files that already exist; fall back to patch which overwrites
        logger.debug("git apply failed, retrying with patch -p1 --force")
        out = env.execute({"command": "cd /testbed && patch -p1 --force < /tmp/model.patch 2>&1 || true"})
        logger.debug(f"patch output: {out.get('output', '')}")


def extract_test_files_from_patch(patch_str: str) -> list[str]:
    """Parse unified diff to find test files added or modified."""
    test_files = []
    for match in re.finditer(r"^\+\+\+ b/(.+)$", patch_str, re.MULTILINE):
        path = match.group(1)
        if re.search(r"(test|tests)[^/]*\.py$|/(test|tests)/", path) or re.search(r"def test_", patch_str):
            if path not in test_files:
                test_files.append(path)
    return test_files


_DJANGO_REPOS = {"django/django"}
_SYMPY_REPOS = {"sympy/sympy"}
_SPHINX_REPOS = {"sphinx-doc/sphinx"}
_PYTHON_REPOS = {"swe-bench/humaneval", "nielstron/humaneval_fix"}

_NON_PYTEST_REPOS = _DJANGO_REPOS | _SYMPY_REPOS | _SPHINX_REPOS | _PYTHON_REPOS

# Script written into the container to re-run failing tests under sys.settrace.
# Reads test node IDs from /tmp/settrace_ids.json, captures the 10 innermost
# /testbed/ source frames (non-test, non-third-party) at the first exception
# event per test, writes results to /tmp/settrace_results.json.
_SETTRACE_RUNNER = """\
import sys, json
from pathlib import Path

TEST_IDS = json.loads(Path("/tmp/settrace_ids.json").read_text())

_TESTBED = "/testbed/"
_SKIP = ("site-packages",)
_TEST_PATTERNS = ("test_", "/tests/", "_test.py", "/test/")


def _is_source(fn):
    if not fn or not fn.startswith(_TESTBED):
        return False
    rel = fn[len(_TESTBED):]
    return not any(p in rel for p in _SKIP)


class _Tracer:
    def __init__(self):
        self._stack = []
        self._seen_calls = set()
        self.all_calls = []
        self.frames = None

    def reset(self):
        self._stack = []
        self._seen_calls = set()
        self.all_calls = []
        self.frames = None

    def __call__(self, frame, event, arg):
        fn = frame.f_code.co_filename
        if event == "call":
            if _is_source(fn):
                rel = fn[len(_TESTBED):]
                info = {
                    "file": rel,
                    "line": frame.f_lineno,
                    "func": frame.f_code.co_name,
                    "is_test": any(p in rel for p in _TEST_PATTERNS),
                    "call_depth": len(self._stack),
                }
                self._stack.append((id(frame), info))
                key = (rel, frame.f_code.co_name)
                if key not in self._seen_calls:
                    self._seen_calls.add(key)
                    self.all_calls.append(info)
        elif event == "return":
            if _is_source(fn) and self._stack and self._stack[-1][0] == id(frame):
                self._stack.pop()
        elif event == "exception" and self._stack:
            frames = [info for _, info in self._stack]
            # Prefer exceptions that originate in test/user code over internal
            # framework exceptions (e.g. pytest marker evaluation fires first).
            has_test = any(info["is_test"] for info in frames)
            prev_has_test = self.frames is not None and any(info["is_test"] for info in self.frames)
            if self.frames is None or (has_test and not prev_has_test):
                self.frames = frames
        return self


class TracerPlugin:
    def __init__(self):
        self._tracer = _Tracer()
        self.results = {}

    def pytest_runtest_call(self, item):
        self._tracer.reset()
        sys.settrace(self._tracer)

    def pytest_runtest_logreport(self, report):
        sys.settrace(None)
        if report.when == "call" and report.failed:
            self.results[report.nodeid] = {
                "exception_frames": self._tracer.frames or [],
                "all_calls": self._tracer.all_calls,
            }


import pytest

plugin = TracerPlugin()
try:
    pytest.main(["-q", "--tb=no", "--noconftest", "--override-ini=addopts="] + TEST_IDS, plugins=[plugin])
except Exception as e:
    sys.stderr.write(f"settrace pytest.main raised: {e}\\n")
finally:
    Path("/tmp/settrace_results.json").write_text(json.dumps(plugin.results, indent=2))
"""

# Django variant: patches unittest.TestCase.run so the tracer works with Django's
# own test runner (runtests.py). Reads dotted test IDs from /tmp/settrace_ids.json,
# derives the module names to pass to runtests.py, and records frames for each
# failing test via an atexit handler.
_DJANGO_SETTRACE_RUNNER = """\
import sys, json, unittest, atexit, runpy
from pathlib import Path

failing_ids = json.loads(Path("/tmp/settrace_ids.json").read_text())
# Each ID is "module.Class.method"; the Django module is everything except the last two parts.
modules = list({".".join(tid.split(".")[:-2]) for tid in failing_ids if tid.count(".") >= 2})

sys.path.insert(0, "/testbed/tests")

_TESTBED = "/testbed/"
_SKIP = ("site-packages",)
_TEST_PATTERNS = ("test_", "/tests/", "_test.py", "/test/")


def _is_source(fn):
    if not fn or not fn.startswith(_TESTBED):
        return False
    rel = fn[len(_TESTBED):]
    return not any(p in rel for p in _SKIP)


class _Tracer:
    def __init__(self):
        self._stack = []
        self._seen_calls = set()
        self.all_calls = []
        self.frames = None

    def reset(self):
        self._stack = []
        self._seen_calls = set()
        self.all_calls = []
        self.frames = None

    def __call__(self, frame, event, arg):
        fn = frame.f_code.co_filename
        if event == "call":
            if _is_source(fn):
                rel = fn[len(_TESTBED):]
                info = {
                    "file": rel,
                    "line": frame.f_lineno,
                    "func": frame.f_code.co_name,
                    "is_test": any(p in rel for p in _TEST_PATTERNS),
                    "call_depth": len(self._stack),
                }
                self._stack.append((id(frame), info))
                key = (rel, frame.f_code.co_name)
                if key not in self._seen_calls:
                    self._seen_calls.add(key)
                    self.all_calls.append(info)
        elif event == "return":
            if _is_source(fn) and self._stack and self._stack[-1][0] == id(frame):
                self._stack.pop()
        elif event == "exception" and self._stack:
            frames = [info for _, info in self._stack]
            has_test = any(info["is_test"] for info in frames)
            prev_has_test = self.frames is not None and any(info["is_test"] for info in self.frames)
            if self.frames is None or (has_test and not prev_has_test):
                self.frames = frames
        return self


_tracer = _Tracer()
_results = {}
_orig_run = unittest.TestCase.run


def _patched_run(self, result=None):
    _tracer.reset()
    sys.settrace(_tracer)
    n_before = (len(result.failures) + len(result.errors)) if result else 0
    try:
        _orig_run(self, result)
    finally:
        sys.settrace(None)
        if result is not None and len(result.failures) + len(result.errors) > n_before:
            key = f"{self.__class__.__module__}.{self.__class__.__name__}.{self._testMethodName}"
            _results[key] = {
                "exception_frames": _tracer.frames or [],
                "all_calls": _tracer.all_calls,
            }


unittest.TestCase.run = _patched_run
atexit.register(lambda: Path("/tmp/settrace_results.json").write_text(json.dumps(_results, indent=2)))

sys.argv = ["runtests.py", "--verbosity", "0", "--settings=test_sqlite", "--parallel", "1"] + modules
runpy.run_path("/testbed/tests/runtests.py", run_name="__main__")
"""


# Sphinx variant: identical to _SETTRACE_RUNNER but without --noconftest, because
# sphinx tests rely on fixtures defined in conftest.py (e.g. the `app` fixture via
# @pytest.mark.sphinx). Without conftest those fixtures are missing and tests fail
# during setup rather than call, so the tracer never fires.
_SPHINX_SETTRACE_RUNNER = """\
import sys, json
from pathlib import Path

TEST_IDS = json.loads(Path("/tmp/settrace_ids.json").read_text())

_TESTBED = "/testbed/"
_SKIP = ("site-packages",)
_TEST_PATTERNS = ("test_", "/tests/", "_test.py", "/test/")


def _is_source(fn):
    if not fn or not fn.startswith(_TESTBED):
        return False
    rel = fn[len(_TESTBED):]
    return not any(p in rel for p in _SKIP)


class _Tracer:
    def __init__(self):
        self._stack = []
        self._seen_calls = set()
        self.all_calls = []
        self.frames = None

    def reset(self):
        self._stack = []
        self._seen_calls = set()
        self.all_calls = []
        self.frames = None

    def __call__(self, frame, event, arg):
        fn = frame.f_code.co_filename
        if event == "call":
            if _is_source(fn):
                rel = fn[len(_TESTBED):]
                info = {
                    "file": rel,
                    "line": frame.f_lineno,
                    "func": frame.f_code.co_name,
                    "is_test": any(p in rel for p in _TEST_PATTERNS),
                    "call_depth": len(self._stack),
                }
                self._stack.append((id(frame), info))
                key = (rel, frame.f_code.co_name)
                if key not in self._seen_calls:
                    self._seen_calls.add(key)
                    self.all_calls.append(info)
        elif event == "return":
            if _is_source(fn) and self._stack and self._stack[-1][0] == id(frame):
                self._stack.pop()
        elif event == "exception" and self._stack:
            frames = [info for _, info in self._stack]
            has_test = any(info["is_test"] for info in frames)
            prev_has_test = self.frames is not None and any(info["is_test"] for info in self.frames)
            if self.frames is None or (has_test and not prev_has_test):
                self.frames = frames
        return self


class TracerPlugin:
    def __init__(self):
        self._tracer = _Tracer()
        self.results = {}

    def pytest_runtest_call(self, item):
        self._tracer.reset()
        sys.settrace(self._tracer)

    def pytest_runtest_logreport(self, report):
        sys.settrace(None)
        if report.when == "call" and report.failed:
            self.results[report.nodeid] = {
                "exception_frames": self._tracer.frames or [],
                "all_calls": self._tracer.all_calls,
            }


import pytest

plugin = TracerPlugin()
try:
    pytest.main(["-q", "--tb=no", "--override-ini=addopts="] + TEST_IDS, plugins=[plugin])
except Exception as e:
    sys.stderr.write(f"settrace pytest.main raised: {e}\\n")
finally:
    Path("/tmp/settrace_results.json").write_text(json.dumps(plugin.results, indent=2))
"""

# Sympy variant: imports each failing test function directly from its module and
# runs it under sys.settrace. Sympy uses its own bin/test runner (no pytest),
# so we cannot use pytest.main here. Instead we parse the failing test IDs from
# the bin/test output, import the module, and call the function directly.
_SYMPY_SETTRACE_RUNNER = """\
import sys, json, importlib
from pathlib import Path

TEST_IDS = json.loads(Path("/tmp/settrace_ids.json").read_text())

_TESTBED = "/testbed/"
_SKIP = ("site-packages",)
_TEST_PATTERNS = ("test_", "/tests/", "_test.py", "/test/")

sys.path.insert(0, "/testbed")


def _is_source(fn):
    if not fn or not fn.startswith(_TESTBED):
        return False
    rel = fn[len(_TESTBED):]
    return not any(p in rel for p in _SKIP)


class _Tracer:
    def __init__(self):
        self._stack = []
        self._seen_calls = set()
        self.all_calls = []
        self.frames = None

    def reset(self):
        self._stack = []
        self._seen_calls = set()
        self.all_calls = []
        self.frames = None

    def __call__(self, frame, event, arg):
        fn = frame.f_code.co_filename
        if event == "call":
            if _is_source(fn):
                rel = fn[len(_TESTBED):]
                info = {
                    "file": rel,
                    "line": frame.f_lineno,
                    "func": frame.f_code.co_name,
                    "is_test": any(p in rel for p in _TEST_PATTERNS),
                    "call_depth": len(self._stack),
                }
                self._stack.append((id(frame), info))
                key = (rel, frame.f_code.co_name)
                if key not in self._seen_calls:
                    self._seen_calls.add(key)
                    self.all_calls.append(info)
        elif event == "return":
            if _is_source(fn) and self._stack and self._stack[-1][0] == id(frame):
                self._stack.pop()
        elif event == "exception" and self._stack:
            frames = [info for _, info in self._stack]
            has_test = any(info["is_test"] for info in frames)
            prev_has_test = self.frames is not None and any(info["is_test"] for info in self.frames)
            if self.frames is None or (has_test and not prev_has_test):
                self.frames = frames
        return self


results = {}
tracer = _Tracer()

for test_id in TEST_IDS:
    # test_id format: "sympy/printing/tests/test_ccode.py::test_ccode_sinc"
    try:
        file_part, func_name = test_id.rsplit("::", 1)
        module_name = file_part[:-3].replace("/", ".")  # strip .py, convert path to dotted module
        module = importlib.import_module(module_name)
        test_func = getattr(module, func_name, None)
        if not callable(test_func):
            results[test_id] = {"exception_frames": [], "all_calls": []}
            continue

        tracer.reset()
        sys.settrace(tracer)
        try:
            test_func()
        except Exception:
            pass
        finally:
            sys.settrace(None)

        results[test_id] = {
            "exception_frames": tracer.frames or [],
            "all_calls": tracer.all_calls,
        }
    except Exception as e:
        sys.stderr.write(f"Error running {test_id}: {e}\\n")
        results[test_id] = {"exception_frames": [], "all_calls": []}

Path("/tmp/settrace_results.json").write_text(json.dumps(results, indent=2))
"""


def _extract_failing_test_ids_pytest(raw_output: str) -> list[str]:
    """Parse pytest FAILED lines into node IDs (requires '::' separator)."""
    ids = []
    for match in re.finditer(r"^FAILED (.+?)(?:\s+-\s+.*)?$", raw_output, re.MULTILINE):
        node_id = match.group(1).strip()
        if "::" in node_id:
            ids.append(node_id)
    return ids


def _extract_failing_test_ids_sympy(raw_output: str) -> list[str]:
    """Parse sympy bin/test failure headers into file::func IDs.

    The actual format is a line of underscores followed by the test path on the next line:
        ________________________________________________________________________________
         sympy/path/to/test.py:func_name
    """
    ids = []
    for match in re.finditer(r"_{5,}\n\s+(\S+\.py):(\w+)\s*\n", raw_output):
        ids.append(f"{match.group(1)}::{match.group(2)}")
    return ids


def _extract_failing_test_ids_django(raw_output: str) -> list[str]:
    """Parse Django/unittest FAIL:/ERROR: lines into dotted IDs (module.Class.method)."""
    ids = []
    for match in re.finditer(r"^(?:FAIL|ERROR): (\w+) \(([^)]+)\)", raw_output, re.MULTILINE):
        method, class_path = match.group(1), match.group(2)
        ids.append(f"{class_path}.{method}")
    return ids


_CONDA_PYTHON = "conda run -n testbed python"


def _run_settrace_pytest(env: Environment, failing_ids: list[str]) -> dict:
    ids_json = json.dumps(failing_ids)
    env.execute({"command": "cat > /tmp/settrace_ids.json << 'SETTRACE_IDS_EOF'\n" + ids_json + "\nSETTRACE_IDS_EOF"})
    env.execute({"command": "cat > /tmp/settrace_runner.py << 'SETTRACE_RUNNER_EOF'\n" + _SETTRACE_RUNNER + "\nSETTRACE_RUNNER_EOF"})
    env.execute({"command": f"cd /testbed && {_CONDA_PYTHON} /tmp/settrace_runner.py 2>/tmp/settrace_errors.log || true"})
    err = env.execute({"command": "cat /tmp/settrace_errors.log 2>/dev/null || true"})
    if err_text := err.get("output", "").strip():
        logger.warning(f"settrace runner stderr: {err_text[:1000]}")
    out = env.execute({"command": "cat /tmp/settrace_results.json 2>/dev/null || echo '{}'"})
    try:
        return json.loads(out.get("output", "{}"))
    except json.JSONDecodeError:
        return {}


def _run_settrace_django(env: Environment, failing_ids: list[str]) -> dict:
    ids_json = json.dumps(failing_ids)
    env.execute({"command": "cat > /tmp/settrace_ids.json << 'SETTRACE_IDS_EOF'\n" + ids_json + "\nSETTRACE_IDS_EOF"})
    env.execute({"command": "cat > /tmp/settrace_runner.py << 'SETTRACE_RUNNER_EOF'\n" + _DJANGO_SETTRACE_RUNNER + "\nSETTRACE_RUNNER_EOF"})
    env.execute({"command": f"cd /testbed && {_CONDA_PYTHON} /tmp/settrace_runner.py 2>/dev/null || true"})
    out = env.execute({"command": "cat /tmp/settrace_results.json 2>/dev/null || echo '{}'"})
    try:
        return json.loads(out.get("output", "{}"))
    except json.JSONDecodeError:
        return {}


def _run_settrace_sphinx(env: Environment, failing_ids: list[str]) -> dict:
    ids_json = json.dumps(failing_ids)
    env.execute({"command": "cat > /tmp/settrace_ids.json << 'SETTRACE_IDS_EOF'\n" + ids_json + "\nSETTRACE_IDS_EOF"})
    env.execute({"command": "cat > /tmp/settrace_runner.py << 'SETTRACE_RUNNER_EOF'\n" + _SPHINX_SETTRACE_RUNNER + "\nSETTRACE_RUNNER_EOF"})
    env.execute({"command": f"cd /testbed && {_CONDA_PYTHON} /tmp/settrace_runner.py 2>/tmp/settrace_errors.log || true"})
    err = env.execute({"command": "cat /tmp/settrace_errors.log 2>/dev/null || true"})
    if err_text := err.get("output", "").strip():
        logger.warning(f"settrace runner stderr: {err_text[:1000]}")
    out = env.execute({"command": "cat /tmp/settrace_results.json 2>/dev/null || echo '{}'"})
    try:
        return json.loads(out.get("output", "{}"))
    except json.JSONDecodeError:
        return {}


def _run_settrace_sympy(env: Environment, failing_ids: list[str]) -> dict:
    ids_json = json.dumps(failing_ids)
    env.execute({"command": "cat > /tmp/settrace_ids.json << 'SETTRACE_IDS_EOF'\n" + ids_json + "\nSETTRACE_IDS_EOF"})
    env.execute({"command": "cat > /tmp/settrace_runner.py << 'SETTRACE_RUNNER_EOF'\n" + _SYMPY_SETTRACE_RUNNER + "\nSETTRACE_RUNNER_EOF"})
    env.execute({"command": f"cd /testbed && {_CONDA_PYTHON} /tmp/settrace_runner.py 2>/tmp/settrace_errors.log || true"})
    err = env.execute({"command": "cat /tmp/settrace_errors.log 2>/dev/null || true"})
    if err_text := err.get("output", "").strip():
        logger.warning(f"settrace runner stderr: {err_text[:1000]}")
    out = env.execute({"command": "cat /tmp/settrace_results.json 2>/dev/null || echo '{}'"})
    try:
        return json.loads(out.get("output", "{}"))
    except json.JSONDecodeError:
        return {}


def run_settrace_on_failing_tests(
    env: Environment, raw_output: str, repo_id: str, test_files: list[str] | None = None
) -> dict:
    """Dispatch settrace to the right runner and return per-test innermost source frames."""
    if repo_id in _DJANGO_REPOS:
        failing_ids = _extract_failing_test_ids_django(raw_output)
        return _run_settrace_django(env, failing_ids) if failing_ids else {}
    if repo_id in _SYMPY_REPOS:
        failing_ids = _extract_failing_test_ids_sympy(raw_output)
        return _run_settrace_sympy(env, failing_ids) if failing_ids else {}
    if repo_id in _PYTHON_REPOS:
        return {}  # custom runners not supported
    if repo_id in _SPHINX_REPOS:
        # Sphinx tests need conftest.py for the `app` fixture, so use the sphinx-specific
        # runner that omits --noconftest (unlike the default pytest runner).
        failing_ids = _extract_failing_test_ids_pytest(raw_output)
        return _run_settrace_sphinx(env, failing_ids) if failing_ids else {}
    failing_ids = _extract_failing_test_ids_pytest(raw_output)
    return _run_settrace_pytest(env, failing_ids) if failing_ids else {}


def _django_paths_to_modules(test_files: list[str]) -> list[str]:
    """Convert file paths like tests/auth/test_basic.py to Django module notation auth.test_basic."""
    modules = []
    for f in test_files:
        # Strip leading tests/ prefix if present
        mod = re.sub(r"^tests/", "", f)
        mod = mod.removesuffix(".py").replace("/", ".")
        modules.append(mod)
    return modules


def get_test_command(repo: str, version: str, test_files: list[str]) -> str | None:
    """Return the shell command to run the given test files for the repo, or None if unknown."""
    _pytest = f"{_CONDA_PYTHON} -m pytest"

    if repo in _DJANGO_REPOS:
        modules = _django_paths_to_modules(test_files)
        if not modules:
            return None
        modules_arg = " ".join(modules)
        return f"cd /testbed && {_CONDA_PYTHON} ./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 {modules_arg} 2>&1 || true"

    if repo in _SYMPY_REPOS:
        files_arg = " ".join(test_files)
        return f"cd /testbed && PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' {_CONDA_PYTHON} bin/test -C --verbose {files_arg} 2>&1 || true"

    if repo in _SPHINX_REPOS:
        files_arg = " ".join(f"/testbed/{f}" for f in test_files)
        return f"cd /testbed && {_pytest} --tb=long -r f --color=no {files_arg} 2>&1 || true"

    if repo in _PYTHON_REPOS:
        cmds = " && ".join(f"{_CONDA_PYTHON} /testbed/{f}" for f in test_files)
        return f"{cmds} 2>&1 || true"

    # Default: pytest (covers astropy, matplotlib, scikit-learn, requests, xarray, etc.)
    # --tb=long: full tracebacks; -r f: force FAILED summary lines even if setup.cfg uses -q;
    # --color=no: prevent ANSI escape codes from breaking the FAILED-line regex;
    # no -x so all failures are collected for settrace extraction.
    files_arg = " ".join(f"/testbed/{f}" for f in test_files)
    return f"cd /testbed && {_pytest} --tb=long -r f --color=no {files_arg} 2>&1 || true"


def run_tests_in_env(env: Environment, test_files: list[str], repo: str = "", version: str = "") -> str:
    """Run tests with verbose tracebacks on the given test files, return raw output."""
    cmd = get_test_command(repo, version, test_files)
    if cmd is None:
        logger.warning(f"No test command available for repo='{repo}' version='{version}'")
        return ""
    out = env.execute({"command": cmd})
    return out.get("output", "")


def extract_stack_traces(raw_output: str) -> list[str]:
    """Extract pytest failure blocks from raw pytest output.

    Matches sections delimited by '_____ test_name _____' headers. Each block
    already contains the full traceback, so a separate traceback pass is not needed
    (and a naive Traceback regex without a correct lookahead produces one giant match).
    """
    return re.findall(r"(_{5,} [^\n]+ _{5,}.*?)(?=\n={3,}|\Z)", raw_output, re.DOTALL)


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    max_instances: int = typer.Option(0, "--max-instances", help="Maximum number of instances to process (0 means no limit)", rich_help_panel="Data selection"),
    instance_ids: str = typer.Option("", "--instance-ids", help="Comma-separated instance_id values to run", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    # When main is called directly (not via CLI), Typer parameters are OptionInfo objects.
    if hasattr(max_instances, "default"):
        max_instances = max_instances.default
    if hasattr(instance_ids, "default"):
        instance_ids = instance_ids.default

    instances = filter_instances(
        instances,
        filter_spec=filter_spec,
        slice_spec=slice_spec,
        shuffle=shuffle,
        limit=max_instances,
    )
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]

    if instance_ids:
        instances = select_instances(instances, instance_ids=instance_ids)

    logger.info(f"Running on {len(instances)} instances...")

    logger.info(f"Building agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    })
    config = recursive_merge(*configs)

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, instance, output_path, config, progress_manager): instance[
                    "instance_id"
                ]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)


if __name__ == "__main__":
    app()
