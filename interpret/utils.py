# interpret/utils.py: helpers of the interpretability rig.
# write_dashboard builds the self-contained dashboard.html of a run: its settings, and per method the plots and the README parts as toggles.

import base64
import html
import json
import re
from pathlib import Path

import markdown

# Style of the dashboard: the browser's own light/dark colors, a readable column, plots side by side.
STYLE = """
:root { color-scheme: light dark; }
body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
table { border-collapse: collapse; }
th, td { border: 1px solid #8884; padding: .3rem .6rem; text-align: left; vertical-align: top; }
details { border: 1px solid #8884; border-radius: 6px; padding: .4rem .8rem; margin: .5rem 0; overflow-x: auto; }
summary { cursor: pointer; font-weight: 600; }
.plots { display: flex; flex-wrap: wrap; gap: 1rem; }
.plots img { max-width: 48%; background: white; }
pre { overflow-x: auto; }
"""


# Writes results_dir/dashboard.html, to open with a double click: the settings of the run, then one section per method with its plots (embedded in the file,
# so it can be moved or sent around) and the parts of that method's subsection of README.md (papers, questions, notes, ...) as toggles.
def write_dashboard(results_dir, data_dir, cfg):
    icfg = cfg["interpret"]
    metadata = json.loads((data_dir / "metadata.json").read_text())
    readme = (Path(__file__).resolve().parent / "README.md").read_text()

    summary = {
        "checkpoint": icfg["checkpoint_path"], "weights": icfg["model_weights"], "dataset": icfg["dataset"],
        "examples": icfg["extract"]["num_samples"], "seed": icfg["seed"],
        "kept layers": ", ".join(metadata["kept_layers"]), "extraction git commit": metadata["git_commit"],
    }
    summary_rows = ""
    for key, value in summary.items():
        summary_rows += f"<tr><th>{key}</th><td>{html.escape(str(value))}</td></tr>"
    body = f"<h1>Interpretability run</h1><table>{summary_rows}</table>"
    body += f"<details><summary>full config</summary><pre>{html.escape(json.dumps(cfg, indent=2))}</pre></details>"

    for name in icfg["methods"]:
        body += f"<h2>{name}</h2><div class='plots'>"
        for plot_path in sorted((results_dir / name).glob("*.png")):
            encoded = base64.b64encode(plot_path.read_bytes()).decode()
            body += f"<img src='data:image/png;base64,{encoded}' alt='{plot_path.stem}'>"
        body += "</div>"

        # this method's subsection of README.md, split at the lines that start one of its parts
        method_text = re.search(rf"^### {name}\n(.*?)(?=^##|\Z)", readme, re.S | re.M).group(1)
        for part in re.split(r"(?m)^(?=Papers|Question|What the paper improves on|Notes on the implementation)", method_text):
            if part.strip() == "":
                continue
            title, _, part_body = part.partition(":")
            body += f"<details><summary>{title}</summary>{markdown.markdown(part_body, extensions=['tables'])}</details>"

    page = f"<!doctype html><html><head><meta charset='utf-8'><title>Interpretability run</title><style>{STYLE}</style></head><body>{body}</body></html>"
    (results_dir / "dashboard.html").write_text(page)
