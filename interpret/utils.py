# interpret/utils.py: helpers of the interpretability rig.
# standardize puts a batch of vectors on a common scale. write_dashboard builds the self-contained dashboard.html of a run: its settings, and per method the plots and the README parts as toggles.

import base64
import html
import json
import re
from pathlib import Path

import markdown

# Style of the dashboard: the browser's own light/dark colors, a readable column, plots side by side.
# TIMING (debug): seconds spent in the dataset's loader path (summed over examples and batches), read and reset by the probe after every training; the getitem parts run in the workers, so they only add up with num_workers 0
TIMINGS = {"getitem tensors": 0.0, "getitem labels": 0.0, "device move": 0.0}

STYLE = """
:root { color-scheme: light dark; }
body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; line-height: 1.55; }
h1 { margin-bottom: 1rem; }
h2 { margin: 0 0 1rem; padding-bottom: .5rem; border-bottom: 2px solid #8886; font-size: 1.5rem; }
table { border-collapse: collapse; }
th, td { border: 1px solid #8884; padding: .3rem .6rem; text-align: left; vertical-align: top; }
tr:nth-child(even) td { background: #8881; }
.card { background: #8881; border: 1px solid #8883; border-radius: 12px; padding: 1.2rem 1.5rem; margin: 2.5rem 0; box-shadow: 0 2px 8px #0002; }
details { border: 1px solid #8884; border-radius: 8px; padding: .4rem .8rem; margin: .6rem 0; overflow-x: auto; }
details[open] { background: #8881; }
summary { cursor: pointer; font-weight: 600; }
summary:hover { opacity: .7; }
.plots { display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem; }
.plots img { max-width: 48%; background: white; border-radius: 8px; box-shadow: 0 1px 4px #0003; }
pre { overflow-x: auto; }
"""


# standardizes (vectorized). 
def standardize(inputs, mean, std):
    return (inputs - mean) / std



# Writes results_dir/dashboard.html, to open with a double click: the settings of the run, then one section per method with its plots (embedded in the file,
# so it can be moved or sent around) and the parts of that method's subsection of README.md (papers, questions, notes, ...) as toggles.
def write_dashboard(results_dir, data_dir, cfg):
    icfg = cfg["interpret"]
    info = json.loads((data_dir / "info.json").read_text())
    readme = (Path(__file__).resolve().parent / "README.md").read_text()

    summary = {
        "checkpoint": icfg["checkpoint_path"], "weights": icfg["model_weights"], "dataset": icfg["dataset"],
        "examples": info["num_samples"], "seed": icfg["seed"],
        "kept layers": ", ".join(info["kept_layers"]), "extraction git commit": info["git_commit"],
    }
    summary_rows = ""
    for key, value in summary.items():
        summary_rows += f"<tr><th>{key}</th><td>{html.escape(str(value))}</td></tr>"
    body = f"<h1>Interpretability run</h1><table>{summary_rows}</table>"
    body += f"<details><summary>full config</summary><pre>{html.escape(json.dumps(cfg, indent=2))}</pre></details>"

    for name in icfg["methods"]:
        headline_plots = ""
        debug_plots = ""
        for plot_path in sorted((results_dir / name).glob("*.png")):
            encoded = base64.b64encode(plot_path.read_bytes()).decode()
            image = f"<img src='data:image/png;base64,{encoded}' alt='{plot_path.stem}'>"
            if plot_path.name.startswith("debug_"):  # the debugging plots of a method are folded away
                debug_plots += image
            else:
                headline_plots += image
        body += f"<section class='card'><h2>{name}</h2><div class='plots'>{headline_plots}</div>"
        if debug_plots != "":
            body += f"<details><summary>debug plots</summary><div class='plots'>{debug_plots}</div></details>"

        # this method's subsection of README.md, split at the lines that start one of its parts
        method_text = re.search(rf"^### {name}\n(.*?)(?=^##|\Z)", readme, re.S | re.M).group(1)
        for part in re.split(r"(?m)^(?=Papers|Question|What the paper improves on|Notes on the implementation)", method_text):
            if part.strip() == "":
                continue
            title, _, part_body = part.partition(":")
            body += f"<details><summary>{title}</summary>{markdown.markdown(part_body, extensions=['tables'])}</details>"
        body += "</section>"

    page = f"<!doctype html><html><head><meta charset='utf-8'><title>Interpretability run</title><style>{STYLE}</style></head><body>{body}</body></html>"
    (results_dir / "dashboard.html").write_text(page)
