
# TraceCompass

TraceCompass is the natural extension of KGCompass, incorporating dynamic traces to inform better fault localization.




## Run Test Generation Locally

Clone the project

```bash
  git clone https://github.com/arvmarandi/trace-compass.git
```

Follow mini-swe-agent's setup guide: https://mini-swe-agent.com/latest/quickstart/

Generate tests

```bash
  mini-extra swebench \                                                                  
  --model [model_provider\model_name] \    
  -o [output_directory] \
  --split test \
  --workers [num_workers]
```

Convert generated tests to a SWT-Bench-compatible format

```bash
  python3 scripts\json_cleaner.py
```

Go to the swt-bench directory

```bash
  cd swt-bench
```

Install dependencies

```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -e .
```

Run evaluations

```bash
  python -m src.main \                                                                                                                              
    --predictions_path ../mini-swe-agent/outputs/swt_bench_compatible.json \
    --filter_swt \
    --max_workers [num_workers] \
    --run_id [iteration_num]
```

The evaluation_results directory will contain all evaluation metrics.
