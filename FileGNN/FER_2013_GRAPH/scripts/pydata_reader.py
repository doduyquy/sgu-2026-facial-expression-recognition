import sys
from pathlib import Path
import html

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from graph.structures import PixelGraph


def esc(x):
    return html.escape(str(x))


def html_table(headers, rows, title=None):
    parts = []
    if title:
        parts.append(f"<h3>{esc(title)}</h3>")
    parts.append("""
    <table style="
        border-collapse: collapse;
        width: 100%;
        font-family: Arial, sans-serif;
        font-size: 13px;
        margin-bottom: 16px;">
    """)
    parts.append("<thead><tr>")
    for h in headers:
        parts.append(
            f"<th style='border:1px solid #ccc;padding:8px;background:#f5f5f5;text-align:left;'>{esc(h)}</th>"
        )
    parts.append("</tr></thead><tbody>")

    for row in rows:
        parts.append("<tr>")
        for cell in row:
            parts.append(
                f"<td style='border:1px solid #ccc;padding:8px;vertical-align:top;'>{esc(cell)}</td>"
            )
        parts.append("</tr>")

    parts.append("</tbody></table>")
    return "".join(parts)


def np_preview(arr, max_rows=8):
    if arr is None:
        return "None"
    arr = np.asarray(arr)
    preview = arr[:max_rows]
    return np.array2string(preview, precision=4, suppress_small=True)


def render_graph_list(data):
    print("<h1 style='font-family:Arial,sans-serif;'>train_graphs.pt</h1>")
    print(f"<p><b>Type:</b> {esc(type(data))}</p>")
    print(f"<p><b>Length:</b> {len(data)}</p>")

    summary_rows = []
    for g in data:
        summary_rows.append([
            g.graph_id,
            g.label,
            g.split,
            g.usage,
            g.height,
            g.width,
            g.node_features.shape[0],
            g.node_features.shape[1],
            g.edge_index.shape[1],
            g.edge_attr.shape[1],
            g.image is None,
        ])

    print(html_table(
        headers=[
            "graph_id", "label", "split", "usage",
            "height", "width",
            "num_nodes", "node_feat_dim",
            "num_edges", "edge_feat_dim",
            "image_is_none"
        ],
        rows=summary_rows,
        title="Summary toàn bộ graph"
    ))

    if not data:
        return

    g = data[0]

    print("<h2 style='font-family:Arial,sans-serif;'>Graph đầu tiên</h2>")

    field_rows = []
    for k, v in vars(g).items():
        if hasattr(v, "shape"):
            field_rows.append([k, type(v).__name__, tuple(v.shape)])
        else:
            field_rows.append([k, type(v).__name__, v])

    print(html_table(
        headers=["Field", "Type", "Value / Shape"],
        rows=field_rows,
        title="Fields"
    ))

    if isinstance(g.metadata, dict):
        meta_rows = [[k, v] for k, v in g.metadata.items()]
        print(html_table(
            headers=["Metadata key", "Value"],
            rows=meta_rows,
            title="Metadata"
        ))

    print("<h3>Preview node_features[:8]</h3>")
    print(f"""
    <pre style="background:#f7f7f7;padding:12px;border:1px solid #ddd;overflow:auto;">
{esc(np_preview(g.node_features, 8))}
    </pre>
    """)

    print("<h3>Preview edge_index[:, :12]</h3>")
    print(f"""
    <pre style="background:#f7f7f7;padding:12px;border:1px solid #ddd;overflow:auto;">
{esc(np.array2string(g.edge_index[:, :12], suppress_small=True))}
    </pre>
    """)

    print("<h3>Preview edge_attr[:12]</h3>")
    print(f"""
    <pre style="background:#f7f7f7;padding:12px;border:1px solid #ddd;overflow:auto;">
{esc(np_preview(g.edge_attr, 12))}
    </pre>
    """)


def process_torch_file(file_path: str):
    data = torch.load(file_path, map_location="cpu", weights_only=False)

    if isinstance(data, list):
        render_graph_list(data)
    elif isinstance(data, dict):
        rows = [[k, type(v).__name__] for k, v in data.items()]
        print("<h1>Dictionary object</h1>")
        print(html_table(["Key", "Type"], rows))
    else:
        print("<h1>Object preview</h1>")
        print(f"<pre>{esc(data)}</pre>")


if __name__ == "__main__":
    file_type = int(sys.argv[1])
    file_path = sys.argv[2]
    process_torch_file(file_path)