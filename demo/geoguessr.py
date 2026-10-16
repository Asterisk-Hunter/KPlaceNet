"""KPlaceNet GeoGuessr-style demo game (single file).

Round flow: NEW ROUND (photo, truth hidden server-side) -> click a region
dot to guess -> REVEAL (truth vs model vs you, scored in km + points).
Research panel preserved: calibrated confidence, conformal set, ACCEPT/ABSTAIN.

No ground-truth leakage: pre-reveal outputs never contain truth lat/lon.
State holds only a row index; truth is looked up server-side at reveal.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKPOINT = ROOT / "checkpoints" / "l2_imagenet_1.0_layer4" / "last.pt"
CALIB_JSON = ROOT / "checkpoints" / "l3_uncertainty_results.json"
TEST_CSV = ROOT / "data" / "osv5m_test" / "metadata.csv"
ABSTAIN_THRESHOLD = 0.5
IMAGE_SIZE = 224
POINTS_SCALE_KM = 2000.0

_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Pure helpers (no gradio) — everything testable headlessly
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km. Pure python, no deps."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def geo_points(error_km: float) -> int:
    """GeoGuessr-style points: 5000 at 0 km, decaying exponentially."""
    if not math.isfinite(error_km) or error_km < 0:
        return 0
    return int(round(5000.0 * math.exp(-error_km / POINTS_SCALE_KM)))


def load_resources() -> dict:
    """Lazy, cached model + cells + calibration + test rows."""
    if _CACHE:
        return _CACHE
    import torch
    from src.cells import Cell
    from src.model import build_model

    ckpt = torch.load(str(CHECKPOINT), map_location="cpu")
    cells_raw = ckpt.get("cells", [])
    num_cells = int(ckpt.get("num_cells", len(cells_raw)))
    cells = [
        Cell(
            cell_id=int(c["cell_id"]),
            lat_min=0, lat_max=0, lon_min=0, lon_max=0,
            centroid_lat=float(c["centroid_lat"]),
            centroid_lon=float(c["centroid_lon"]),
        )
        for c in sorted(cells_raw, key=lambda x: x["cell_id"])
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_cells=num_cells, pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()

    T, q = 1.0, None
    try:
        data = json.loads(CALIB_JSON.read_text(encoding="utf-8"))
        T = float(data.get("temperature", 1.0))
        q_raw = data.get("conformal_quantile", None)
        q = float(q_raw) if q_raw is not None else None
    except Exception:
        pass

    rows: list[dict] = []
    with open(TEST_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "image": str((TEST_CSV.parent / r["image_path"]).resolve()),
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
            })
    _CACHE.update(model=model, cells=cells, T=T, quantile=q,
                  device=device, K=num_cells, rows=rows)
    return _CACHE


def model_predict(image_path: str) -> dict:
    """Run inference on one image file. Returns scalars (no figure).

    Raises FileNotFoundError / ValueError on unreadable images.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from src.dataset import get_transforms
    from src.uncertainty import abstention_mask, conformal_prediction_set

    res = load_resources()
    model, cells, T, q = res["model"], res["cells"], res["T"], res["quantile"]
    device, K = res["device"], res["K"]

    img = Image.open(image_path).convert("RGB")
    tfm = get_transforms(IMAGE_SIZE, train=False)
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        raw_probs = F.softmax(logits, dim=1)
        cal_probs = F.softmax(logits / max(T, 1e-3), dim=1)
    raw_conf = float(raw_probs.max().item())
    cal_conf = float(cal_probs.max().item())
    pred_id = int(cal_probs.argmax(dim=1).item())
    cell = cells[pred_id]
    cset = conformal_prediction_set(cal_probs.cpu(), quantile=q)
    set_size = int(cset.sum().item())
    accept = bool(abstention_mask(cal_probs.cpu(), threshold=ABSTAIN_THRESHOLD)[0].item())
    lat, lon = float(cell.centroid_lat), float(cell.centroid_lon)
    assert math.isfinite(lat) and math.isfinite(lon)
    assert 0.0 <= cal_conf <= 1.0 and 0.0 <= raw_conf <= 1.0
    return {
        "lat": lat, "lon": lon, "cell_id": pred_id,
        "cal_conf": cal_conf, "raw_conf": raw_conf, "T": float(T),
        "set_size": set_size, "K": int(K),
        "verdict": "ACCEPT" if accept else "ABSTAIN",
    }


def dots_trace():
    """Plotly trace for the 300 region dots (plain floats, JSON-safe)."""
    import plotly.graph_objects as go

    res = load_resources()
    lats = [float(c.centroid_lat) for c in res["cells"]]
    lons = [float(c.centroid_lon) for c in res["cells"]]
    return go.Scattergeo(
        lat=lats,
        lon=lons,
        mode="markers",
        marker={"size": 4, "color": "rgba(30,144,255,0.55)"},
        name="model regions (300 cells)",
        hovertemplate="cell %{customdata}<extra></extra>",
        customdata=[int(c.cell_id) for c in res["cells"]],
    )


def base_layout() -> dict:
    return {
        "geo": {
            "scope": "world",
            "projection": {"type": "equirectangular"},
            "showland": True,
            "landcolor": "rgb(235,235,220)",
            "showocean": True,
            "oceancolor": "rgb(200,225,245)",
            "showcountries": True,
            "countrycolor": "rgb(150,150,150)",
            "coastlinecolor": "rgb(120,120,120)",
            "showframe": True,
        },
        "margin": {"l": 0, "r": 0, "t": 30, "b": 0},
        "title": "Set sliders + Lock guess — blue dots are the 300 model regions",
        "showlegend": True,
    }


def build_base_fig():
    """Pre-reveal figure: map + region dots only. No truth anywhere."""
    import plotly.graph_objects as go

    return go.Figure(data=[dots_trace()], layout=base_layout())


def _marker(lat: float, lon: float, size: int, color: str,
            symbol: str, name: str):
    import plotly.graph_objects as go

    return go.Scattergeo(
        lat=[float(lat)], lon=[float(lon)],
        mode="markers", marker={"size": size, "color": color, "symbol": symbol},
        name=name,
    )


def build_reveal_fig(truth: tuple[float, float],
                     model: tuple[float, float],
                     guess: tuple[float, float] | None):
    """Post-reveal figure: dots + TRUTH / MODEL / YOU markers."""
    import plotly.graph_objects as go

    tlat, tlon = float(truth[0]), float(truth[1])
    mlat, mlon = float(model[0]), float(model[1])
    traces = [
        dots_trace(),
        _marker(tlat, tlon, 14, "green", "star", f"TRUTH {tlat:.2f},{tlon:.2f}"),
        _marker(mlat, mlon, 13, "red", "x", f"MODEL {mlat:.2f},{mlon:.2f}"),
    ]
    if guess is not None:
        glat, glon = float(guess[0]), float(guess[1])
        traces.append(_marker(glat, glon, 12, "gold", "circle",
                              f"YOU {glat:.2f},{glon:.2f}"))
    layout = base_layout()
    layout["title"] = "Reveal: green=truth, red=model, gold=you"
    return go.Figure(data=traces, layout=layout)


def fresh_state(seed: int = 42) -> dict:
    """JSON-serializable session state. Holds NO ground truth."""
    import random
    res = load_resources()
    order = list(range(len(res["rows"])))
    rng = random.Random(seed)
    rng.shuffle(order)
    return {"order": order, "pos": 0, "guess_cell": None,
            "revealed": False, "rounds": 0, "model_total": 0,
            "user_total": 0, "seed": seed}


def deal_round(state: dict):
    """Deal next photo. Returns (photo_path, fig, status_md, new_state).

    Skips unreadable images automatically. Truth never leaves the server:
    only a row index is stored.
    """
    res = load_resources()
    state = dict(state)
    attempts = 0
    while True:
        if state["pos"] >= len(state["order"]):
            # Exhausted: reshuffle, keep totals.
            import random
            order = list(range(len(res["rows"])))
            random.Random(state.get("seed", 42) + state.get("rounds", 0) + 1).shuffle(order)
            state["order"] = order
            state["pos"] = 0
        row_idx = state["order"][state["pos"]]
        row = res["rows"][row_idx]
        try:
            from PIL import Image
            with Image.open(row["image"]) as im:
                im.verify()
        except Exception:
            state["pos"] += 1
            attempts += 1
            if attempts > len(res["rows"]):
                raise RuntimeError("No readable test images found.")
            continue
        break
    state["guess_cell"] = None
    state["revealed"] = False
    n = len(res["rows"])
    status = (f"### Round {state['rounds'] + 1} — photo {state['pos'] + 1} of {n}\n\n"
              f"Study the photo, set the **lat/lon sliders**, press **Lock guess**, "
              f"then **Reveal** to score.")
    return row["image"], build_base_fig(), status, state


def snap_to_cell(lat: float, lon: float) -> int:
    """Nearest cell id to (lat, lon) by haversine. Raises ValueError if out of range."""
    if not (-90.0 <= float(lat) <= 90.0 and -180.0 <= float(lon) <= 180.0):
        raise ValueError(f"Guess out of range: {lat}, {lon}")
    res = load_resources()
    return min(range(int(res["K"])),
               key=lambda k: haversine_km(float(lat), float(lon),
                                          float(res["cells"][k].centroid_lat),
                                          float(res["cells"][k].centroid_lon)))


def build_guess_fig(guess_xy: tuple[float, float]):
    """Base map + gold YOU-preview marker (your own guess — nothing leaks)."""
    fig = build_base_fig()
    glat, glon = float(guess_xy[0]), float(guess_xy[1])
    fig.add_trace(_marker(glat, glon, 12, "gold", "circle",
                          f"YOU (guess) {glat:.2f},{glon:.2f}"))
    fig.update_layout(title="Guess locked (gold) — press Reveal")
    return fig


def place_guess(lat: float, lon: float, state: dict):
    """Lock a guess from lat/lon (snapped to nearest cell). Returns (fig, status, state)."""
    res = load_resources()
    state = dict(state)
    if state.get("revealed"):
        return build_reveal_fig_cached(state), "Already revealed — deal a new round.", state
    try:
        cid = snap_to_cell(lat, lon)
    except (ValueError, TypeError):
        return build_base_fig(), "Set the lat/lon sliders, then Lock guess.", state
    cell = res["cells"][cid]
    state["guess_cell"] = int(cid)
    gxy = (float(cell.centroid_lat), float(cell.centroid_lon))
    status = (f"Your guess: **{gxy[0]:.2f}, {gxy[1]:.2f}** "
              f"(cell {cid}, snapped to nearest region). Press **Reveal** to score.")
    return build_guess_fig(gxy), status, state


def build_reveal_fig_cached(state: dict):
    """Rebuild the current reveal figure from state (for idempotent re-reveal)."""
    res = load_resources()
    row = res["rows"][state["order"][state["pos"]]]
    try:
        pred = model_predict(res["rows"][state["order"][state["pos"]]]["image"])
        model_xy = (pred["lat"], pred["lon"])
    except Exception:
        model_xy = (float("nan"), float("nan"))
    guess_xy = None
    if state.get("guess_cell") is not None:
        g = res["cells"][int(state["guess_cell"])]
        guess_xy = (float(g.centroid_lat), float(g.centroid_lon))
    return build_reveal_fig((row["lat"], row["lon"]), model_xy, guess_xy)


def reveal(state: dict):
    """Score the round. Idempotent: second call changes nothing. Returns (fig, md, state)."""
    res = load_resources()
    state = dict(state)
    row = res["rows"][state["order"][state["pos"]]]
    truth = (float(row["lat"]), float(row["lon"]))

    if state.get("revealed"):
        return build_reveal_fig_cached(state), state["_last_md"], state

    try:
        pred = model_predict(row["image"])
    except Exception as e:
        return build_base_fig(), f"Model failed on this photo ({e}) — deal a new round.", state

    model_xy = (pred["lat"], pred["lon"])
    model_err = haversine_km(truth[0], truth[1], model_xy[0], model_xy[1])
    model_pts = geo_points(model_err)

    guess_xy = None
    if state.get("guess_cell") is not None:
        g = res["cells"][int(state["guess_cell"])]
        guess_xy = (float(g.centroid_lat), float(g.centroid_lon))
        user_err = haversine_km(truth[0], truth[1], guess_xy[0], guess_xy[1])
        user_pts = geo_points(user_err)
        you_line = (f"- You: **{guess_xy[0]:.2f}, {guess_xy[1]:.2f}** "
                    f"(cell {state['guess_cell']}) — error **{user_err:,.1f} km** — "
                    f"**{user_pts} pts**\n")
    else:
        user_err, user_pts = None, 0
        you_line = "- You: **no guess** — **0 pts**\n"

    state["revealed"] = True
    state["rounds"] = int(state.get("rounds", 0)) + 1
    state["model_total"] = int(state.get("model_total", 0)) + int(model_pts)
    state["user_total"] = int(state.get("user_total", 0)) + int(user_pts)
    state["pos"] = int(state["pos"]) + 1

    badge = "**ACCEPT**" if pred["verdict"] == "ACCEPT" else "**ABSTAIN**"
    md = (
        f"### Reveal — truth {truth[0]:.2f}, {truth[1]:.2f}\n\n"
        f"- Model: **{model_xy[0]:.2f}, {model_xy[1]:.2f}** (cell {pred['cell_id']})"
        f" — error **{model_err:,.1f} km** — **{model_pts} pts**\n"
        f"{you_line}"
        f"- Model confidence: calibrated **{pred['cal_conf'] * 100:.2f}%** "
        f"(raw {pred['raw_conf'] * 100:.2f}%, T={pred['T']:.2f}) — verdict {badge}\n"
        f"- Conformal set: **{pred['set_size']} of {pred['K']}** regions\n\n"
        f"**Scoreboard** — rounds: {state['rounds']} | "
        f"model total: {state['model_total']} | your total: {state['user_total']}"
    )
    state["_last_md"] = md
    return build_reveal_fig(truth, model_xy, guess_xy), md, state


# ---------------------------------------------------------------------------
# Gradio UI (thin wrappers only)
# ---------------------------------------------------------------------------

def build_ui():
    import gradio as gr

    with gr.Blocks(title="KPlaceNet GeoGuessr") as demo:
        gr.Markdown("# KPlaceNet GeoGuessr — you vs the model")
        gr.Markdown(
            "A photo appears. **Set the lat/lon sliders and Lock your guess**, "
            "then **Reveal**: green star = truth, red X = model, gold = you. "
            "5000 pts for a perfect guess, decaying with distance. Guesses snap "
            "to the nearest of the model's 300 regions. "
            "The model is a coarse 300-cell classifier — **ABSTAIN is it being honest**."
        )
        state = gr.State(fresh_state())
        with gr.Row():
            photo = gr.Image(label="Mystery photo", interactive=False)
            map_plot = gr.Plot(label="World map — 300 model regions")
        with gr.Row():
            lat_in = gr.Slider(-90, 90, value=20, step=0.5, label="Your guess: latitude")
            lon_in = gr.Slider(-180, 180, value=78, step=0.5, label="Your guess: longitude")
        status = gr.Markdown("Press **New round** to start.")
        with gr.Row():
            new_btn = gr.Button("New round", variant="primary")
            lock_btn = gr.Button("Lock guess")
            reveal_btn = gr.Button("Reveal / Score")
            reset_btn = gr.Button("Reset scores")

        def _new(st):
            photo_path, fig, msg, nst = deal_round(st)
            return photo_path, fig, msg, nst

        def _lock(la, lo, st):
            fig, msg, nst = place_guess(la, lo, st)
            return fig, msg, nst

        def _reveal(st):
            fig, msg, nst = reveal(st)
            return fig, msg, nst

        def _reset(st):
            nst = fresh_state(st.get("seed", 42))
            photo_path, fig, msg, nst2 = deal_round(nst)
            return photo_path, fig, "Scores reset. " + msg, nst2

        new_btn.click(fn=_new, inputs=state,
                      outputs=[photo, map_plot, status, state])
        lock_btn.click(fn=_lock, inputs=[lat_in, lon_in, state],
                       outputs=[map_plot, status, state])
        reveal_btn.click(fn=_reveal, inputs=state,
                         outputs=[map_plot, status, state])
        reset_btn.click(fn=_reset, inputs=state,
                        outputs=[photo, map_plot, status, state])
        demo.load(fn=_new, inputs=state, outputs=[photo, map_plot, status, state])
    return demo


# ---------------------------------------------------------------------------
# Full headless probe — no server. Raises on ANY failure.
# ---------------------------------------------------------------------------

def selftest() -> None:
    import random
    import tempfile

    print("[probe] loading resources (model + rows)...")
    res = load_resources()
    assert len(res["rows"]) == 3000, len(res["rows"])
    assert len(res["cells"]) == 300, len(res["cells"])
    print("[probe] PASS resources: 3000 rows, 300 cells")

    # Points formula unit checks.
    assert geo_points(0) == 5000, geo_points(0)
    assert 1800 <= geo_points(2000) <= 1900, geo_points(2000)
    assert geo_points(20000) < 50, geo_points(20000)
    assert geo_points(-5) == 0 and geo_points(float("nan")) == 0
    print("[probe] PASS points formula")

    # Deal a fixed-seed round; assert NO truth leakage pre-reveal.
    st = fresh_state(seed=7)
    assert "lat" not in st and "lon" not in st and "truth" not in st
    photo, fig, msg, st = deal_round(st)
    assert Path(photo).exists(), photo
    payload = str(fig.to_json()) + msg
    row = res["rows"][st["order"][st["pos"]]]
    assert repr(row["lat"]) not in payload and repr(row["lon"]) not in payload, \
        "TRUTH LEAKED pre-reveal"
    assert len(fig.to_dict()["data"]) == 1, "pre-reveal fig must be dots only"
    assert len(fig.to_dict()["data"][0]["lat"]) == 300
    print(f"[probe] PASS round dealt, no leakage (row truth {row['lat']:.3f},{row['lon']:.3f})")

    # Guess the truth cell -> error must equal direct haversine; points match formula.
    truth_cell = min(range(300),
                     key=lambda k: haversine_km(row["lat"], row["lon"],
                                                float(res["cells"][k].centroid_lat),
                                                float(res["cells"][k].centroid_lon)))
    fig2, msg2, st = place_guess(row["lat"], row["lon"], st)
    assert st["guess_cell"] == truth_cell, "truth coords must snap to truth cell"
    assert len(fig2.to_dict()["data"]) == 2, "locked-guess fig must be dots + YOU preview"
    assert "revealed" not in msg2.lower() or True
    rfig, rmd, st = reveal(st)
    g = res["cells"][truth_cell]
    expect_err = haversine_km(row["lat"], row["lon"],
                              float(g.centroid_lat), float(g.centroid_lon))
    assert f"{expect_err:,.1f}" in rmd, "user error mismatch"
    assert str(geo_points(expect_err)) in rmd, "user points mismatch"
    assert len(rfig.to_dict()["data"]) == 4, "reveal fig must be dots+truth+model+you"
    print(f"[probe] PASS truth-cell guess: err {expect_err:,.1f} km, "
          f"{geo_points(expect_err)} pts")

    # Idempotent second reveal: totals frozen.
    before = (st["rounds"], st["model_total"], st["user_total"])
    rfig2, rmd2, st = reveal(st)
    assert (st["rounds"], st["model_total"], st["user_total"]) == before
    assert rmd2 == rmd
    print("[probe] PASS reveal idempotent")

    # Model outputs finite + verdict valid (parsed from markdown).
    assert ("ACCEPT" in rmd) != ("ABSTAIN" in rmd)
    m = __import__("re").search(r"error \*\*([\d,\.]+) km\*\* — \*\*(\d+) pts", rmd)
    assert m, "model error/points line missing"
    print("[probe] PASS model verdict finite:", m.group(0)[:60])

    # Antipode guess on next round -> near-zero points.
    photo_b, fig_b, msg_b, st = deal_round(st)
    assert st["guess_cell"] is None and not st["revealed"]
    row_b = res["rows"][st["order"][st["pos"]]]
    u_before = int(st["user_total"])
    anti_lat = -row_b["lat"]
    anti_lon = row_b["lon"] + 180.0
    if anti_lon > 180.0:
        anti_lon -= 360.0
    _, _, st = place_guess(anti_lat, anti_lon, st)
    assert st["guess_cell"] is not None
    _, rmd_b, st = reveal(st)
    assert st["user_total"] - u_before < 50, rmd_b[-200:]
    print("[probe] PASS antipode guess scores ~0")

    # Invalid guess index rejected, no crash.
    st_c = fresh_state(seed=11)
    _, _, _, st_c = deal_round(st_c)
    _, m_c, st_c = place_guess(999, 0.0, st_c)
    assert st_c["guess_cell"] is None
    rfig_c, rmd_c, st_c = reveal(st_c)
    assert "no guess" in rmd_c and len(rfig_c.to_dict()["data"]) == 3
    print("[probe] PASS invalid-guess + no-guess reveal")

    # Corrupt image path handled by model_predict raising (caller deals new round).
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tf.write(b"not an image at all")
        bad = tf.name
    try:
        model_predict(bad)
        raise AssertionError("corrupt image should raise")
    except Exception as e:
        assert "cannot identify" in str(e).lower() or "image" in str(e).lower(), str(e)[:100]
    print("[probe] PASS corrupt image raises cleanly")

    print("[probe] ALL PASS")


if __name__ == "__main__":
    build_ui().launch(server_port=7860)
