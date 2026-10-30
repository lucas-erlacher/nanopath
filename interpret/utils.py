# interpret/utils.py: helpers of the interpretability rig.
# tissue_masks finds the tissue pixels of a batch of tiles. write_dashboard builds the self-contained dashboard.html of a run: its settings, and per method the plots and the README parts as toggles;
# for RELAX, relax_layers_html shows the maps as layers over the image, each with an opacity slider.

import base64
import html
import io
import json
import re
from pathlib import Path

import markdown
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

from prepare import tissue_mask

# Style of the dashboard: the browser's own light/dark colors, a readable column, plots side by side.
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
.relax-image { display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: center; width: 100%; margin-bottom: 1.5rem; }
.layers { position: relative; width: 336px; max-width: 100%; }
.layers img { display: block; width: 100%; max-width: none; background: none; box-shadow: none; border-radius: 0; }
.layers img + img { position: absolute; top: 0; left: 0; }
.sliders label { display: flex; align-items: center; gap: .6rem; margin: .4rem 0; }
.scale { display: inline-block; width: 6rem; height: .7rem; border-radius: 3px; }
"""


# The tissue mask of a batch of tiles with the repo's rule (tissue_mask in prepare.py, which takes one tile at a time). The rule thresholds the saturation of RGB values in [0, 1], so a batch that is normalized, in 0-255
# or without the batch dimension would give a meaningless mask without any error: the preconditions are asserted here, every use of the tissue rule in the rig goes through this function.
def tissue_masks(images):  # (B, 3, H, W) floats in [0, 1] -> (B, H, W) booleans
    assert images.ndim == 4 and images.shape[1] == 3, f"expected tiles of shape (B, 3, H, W), got {tuple(images.shape)}"
    assert images.dtype.is_floating_point, f"expected float tiles in [0, 1], got {images.dtype}"
    assert images.min() >= 0.0 and images.max() <= 1.0, f"the tissue rule needs RGB in [0, 1] (not normalized, not 0-255), got the range [{images.min().item():.3g}, {images.max().item():.3g}]"
    return torch.stack([tissue_mask(tile) for tile in images])


# The RELAX maps for the dashboard, built from its results (not from its static figures): per image, the importance, the uncertainty and the thresholded importance as coolwarm heat maps
# (cold = low, warm = high) stacked over the input, each with an opacity slider and its colour scale with its value range; the uncertain pixels of the thresholded map are transparent.
def relax_layers_html(relax_results):
    colormap = colormaps["coolwarm"]
    gradient_stops = []
    for position in (0.0, 0.25, 0.5, 0.75, 1.0):
        rgba = colormap(position)
        gradient_stops.append(f"rgb({round(rgba[0] * 255)}, {round(rgba[1] * 255)}, {round(rgba[2] * 255)})")
    gradient = f"linear-gradient(to right, {', '.join(gradient_stops)})"

    page_part = ""
    for image_index in relax_results["image_indices"]:
        maps = relax_results["maps"][image_index]
        image_channels_last = maps["image"].permute(1, 2, 0).numpy()  # (H, W, 3) in [0, 1]
        layers = {"image": (image_channels_last * 255).round().astype(np.uint8)}  # {layer name: (H, W, 3 or 4) uint8}
        value_ranges = {}
        for map_name in ("importance", "uncertainty", "thresholded_importance"):
            values = maps[map_name].numpy()  # (H, W)
            is_shown = np.ones_like(values, dtype=bool)  # (H, W), the pixels that are coloured
            if map_name == "thresholded_importance":
                is_shown = values != 0  # the uncertain pixels (set to 0) are not coloured
            low = values[is_shown].min()
            high = values[is_shown].max()
            scaled = (values - low) / (high - low)  # [0, 1] over the shown range of this map
            rgba = colormap(scaled)  # (H, W, 4) in [0, 1]
            rgba[:, :, 3] = is_shown  # transparent where not shown
            layers[map_name] = (rgba * 255).round().astype(np.uint8)
            value_ranges[map_name] = (low, high)

        images = ""
        sliders = ""
        for layer_name, pixels in layers.items():
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode()
            layer_id = f"relax_{image_index}_{layer_name}"
            opacity = 0.0  # the uncertainty and the thresholded importance are switched on with their sliders
            if layer_name == "image":
                opacity = 1.0
            if layer_name == "importance":
                opacity = 0.5  # shown at the start, at half opacity
            images += f"<img id='{layer_id}' src='data:image/png;base64,{encoded}' alt='{layer_name}' style='opacity: {opacity}'>"
            if layer_name != "image":
                low, high = value_ranges[layer_name]
                slider = f"<input type='range' min='0' max='100' value='{round(opacity * 100)}' oninput=\"document.getElementById('{layer_id}').style.opacity = this.value / 100\">"
                sliders += f"<label>{slider} {layer_name.replace('_', ' ')} <span class='scale' style='background: {gradient}'></span> {low:.3g} to {high:.3g}</label>"
        page_part += f"<div class='relax-image'><div class='layers'>{images}</div><div class='sliders'><b>image {image_index}</b>{sliders}</div></div>"
    return page_part


# Writes results_dir/dashboard.html, to open with a double click: the settings of the run, then one section per method with its plots (embedded in the file,
# so it can be moved or sent around) and the parts of that method's subsection of README.md (papers, questions, notes, ...) as toggles.
def write_dashboard(results_dir, data_dir, cfg):
    icfg = cfg["interpret"]
    extraction_info = json.loads((data_dir / "info.json").read_text())
    readme = (Path(__file__).resolve().parent / "README.md").read_text()
    results = torch.load(results_dir / "results.pt", weights_only=False)  # the outputs of all methods (see main() in interpret.py)

    summary = {
        "checkpoint": icfg["checkpoint_path"], "weights": icfg["model_weights"], "dataset": icfg["dataset"],
        "examples": extraction_info["num_samples"], "seed": icfg["seed"],
        "kept layers": ", ".join(extraction_info["kept_layers"]), "extraction git commit": extraction_info["git_commit"],
    }
    summary_rows = ""
    for setting, setting_value in summary.items():
        summary_rows += f"<tr><th>{setting}</th><td>{html.escape(str(setting_value))}</td></tr>"
    body = f"<h1>Interpretability run</h1><table>{summary_rows}</table>"
    body += f"<details><summary>full config</summary><pre>{html.escape(json.dumps(cfg, indent=2))}</pre></details>"

    for method_name in icfg["methods"]:
        headline_plots = ""
        debug_plots = ""
        for plot_path in sorted((results_dir / method_name).glob("*.png")):
            encoded = base64.b64encode(plot_path.read_bytes()).decode()
            image = f"<img src='data:image/png;base64,{encoded}' alt='{plot_path.stem}'>"
            if plot_path.name.startswith("debug_"):  # the debugging plots of a method are folded away
                debug_plots += image
            else:
                headline_plots += image
        if method_name == "relax":  # the maps as layers with opacity sliders instead of the static figures (which stay in its folder)
            headline_plots = relax_layers_html(results["relax"])
        body += f"<section class='card'><h2>{method_name}</h2><div class='plots'>{headline_plots}</div>"
        if debug_plots != "":
            body += f"<details><summary>debug plots</summary><div class='plots'>{debug_plots}</div></details>"

        # this method's subsection of README.md, split at the lines that start one of its parts
        method_text = re.search(rf"^### {method_name}\n(.*?)(?=^##|\Z)", readme, re.S | re.M).group(1)
        for part in re.split(r"(?m)^(?=Papers|Question|What the paper improves on|Notes on the paper)", method_text):
            if part.strip() == "":
                continue
            title, _, part_body = part.partition(":")
            body += f"<details><summary>{title}</summary>{markdown.markdown(part_body, extensions=['tables'])}</details>"
        body += "</section>"

    page = f"<!doctype html><html><head><meta charset='utf-8'><title>Interpretability run</title><style>{STYLE}</style></head><body>{body}</body></html>"
    (results_dir / "dashboard.html").write_text(page)
