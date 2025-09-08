import io
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st


def set_page() -> None:
    """Configure Streamlit page settings and global styles."""
    st.set_page_config(
        page_title="FSLR Placeholder UI",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def pad_or_trim(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Pad with zeros or trim sequence to target_length along time axis.

    Args:
        sequence: Array shaped [T, D].
        target_length: Desired temporal length T.

    Returns:
        Array shaped [target_length, D], float32.
    """
    if sequence.ndim != 2:
        raise ValueError(f"Expected 2D sequence [T, D], got shape {sequence.shape}")

    time_steps, feature_dim = sequence.shape
    sequence = sequence.astype(np.float32)

    if time_steps == target_length:
        return sequence
    if time_steps > target_length:
        return sequence[:target_length]

    output = np.zeros((target_length, feature_dim), dtype=np.float32)
    output[:time_steps] = sequence
    return output


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax for numpy arrays."""
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


def simulate_predictions(
    random_state: np.random.RandomState,
    num_gloss_classes: int,
    num_category_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate simulated logits for gloss and category heads."""
    gloss_logits = random_state.randn(num_gloss_classes).astype(np.float32)
    cat_logits = random_state.randn(num_category_classes).astype(np.float32)
    return gloss_logits, cat_logits


def topk_from_logits(logits: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return top-k indices and probabilities from logits vector."""
    k = max(1, min(k, logits.shape[-1]))
    probs = softmax(logits)
    topk_idx = np.argpartition(-probs, kth=k - 1)[:k]
    topk_sorted = topk_idx[np.argsort(-probs[topk_idx])]
    return topk_sorted, probs[topk_sorted]


def render_sidebar() -> Dict:
    """Render sidebar controls and return configuration dict."""
    st.sidebar.title("FSLR Demo (Placeholder)")
    st.sidebar.caption(
        "Upload a preprocessed .npz (X, mask, timestamps_ms, meta). Predictions are simulated."
    )

    with st.sidebar.expander("Prediction settings", expanded=True):
        model_choice = st.selectbox("Model", ["SignTransformer", "IV3_GRU"], index=0)
        st.caption("Both options are placeholders here; predictions are simulated.")

        sequence_length = st.slider("Sequence length T", min_value=50, max_value=300, value=150, step=10)
        topk = st.slider("Top-K", min_value=1, max_value=10, value=5)

        num_gloss_classes = st.number_input("# Gloss classes", min_value=2, max_value=2000, value=105, step=1)
        num_category_classes = st.number_input("# Category classes", min_value=2, max_value=200, value=10, step=1)

        random_seed = st.number_input("Random seed (simulation)", min_value=0, max_value=1_000_000, value=42, step=1)

    return dict(
        model_choice=model_choice,
        sequence_length=int(sequence_length),
        topk=int(topk),
        num_gloss_classes=int(num_gloss_classes),
        num_category_classes=int(num_category_classes),
        random_seed=int(random_seed),
    )


def render_sequence_overview(npz_dict: Dict, sequence_length: int) -> Tuple[np.ndarray, Dict]:
    """Show metadata and return processed sequence X_pad.

    Args:
        npz_dict: NpzFile-like dict from numpy.load.
        sequence_length: Target T for padding/trimming.

    Returns:
        (X_pad, meta_dict)
    """
    if "X" not in npz_dict:
        raise KeyError("Uploaded .npz must contain key 'X' with shape [T, 156]")

    X_raw = np.array(npz_dict["X"])  # [T, 156]
    mask = np.array(npz_dict.get("mask", []))
    timestamps_ms = np.array(npz_dict.get("timestamps_ms", []))

    raw_length, raw_features = X_raw.shape[0], X_raw.shape[1] if X_raw.ndim == 2 else (0, 0)

    st.subheader("Sequence overview")
    cols = st.columns(4)
    cols[0].metric("Frames (raw)", f"{raw_length}")
    cols[1].metric("Features", f"{raw_features}")
    if timestamps_ms.size > 0:
        duration_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0 if timestamps_ms.size > 1 else 0.0
        cols[2].metric("Duration (s)", f"{duration_s:.2f}")
    cols[3].metric("Target T", f"{sequence_length}")

    meta_raw = npz_dict.get("meta")
    meta_parsed: Dict = {}
    if meta_raw is not None:
        try:
            if isinstance(meta_raw, (str, bytes)):
                meta_parsed = json.loads(meta_raw)
            else:
                meta_parsed = json.loads(str(meta_raw))
        except Exception:
            meta_parsed = {"info": "Unparsed meta"}

    if meta_parsed:
        with st.expander("Metadata", expanded=False):
            st.json(meta_parsed)

    X_pad = pad_or_trim(X_raw, sequence_length)
    return X_pad, meta_parsed


def render_feature_charts(sequence: np.ndarray) -> None:
    """Render simple line charts for a subset of features over time."""
    st.subheader("Feature preview")
    time_steps, feature_dim = sequence.shape

    start_index = st.number_input(
        "Start feature index",
        min_value=0,
        max_value=max(0, feature_dim - 6),
        value=0,
        step=2,
        help="Features are 156-dim (x,y per keypoint). Choose a starting index to preview 6 dims.",
    )
    num_preview_dims = 6
    end_index = int(start_index) + num_preview_dims

    preview_data = sequence[:, int(start_index) : end_index]
    df = pd.DataFrame(preview_data, columns=[f"f_{i}" for i in range(int(start_index), end_index)])
    st.line_chart(df, use_container_width=True)


def render_topk_table(indices: np.ndarray, probs: np.ndarray, label_prefix: str) -> None:
    df = pd.DataFrame(
        {
            "rank": np.arange(1, len(indices) + 1, dtype=int),
            "index": indices,
            "probability": np.round(probs, 4),
            "label": [f"{label_prefix}_{i}" for i in indices],
        }
    )
    st.dataframe(df, hide_index=True, use_container_width=True)


def main() -> None:
    set_page()
    cfg = render_sidebar()

    st.title("Filipino Sign Language Recognition — Placeholder UI")
    st.write(
        "This demo accepts preprocessed .npz files and simulates predictions for gloss and category."
    )

    uploaded_file = st.file_uploader("Upload preprocessed .npz", type=["npz"])

    if uploaded_file is None:
        st.info("Upload a .npz file containing keys: 'X' (required), optionally 'mask', 'timestamps_ms', 'meta'.")
        return

    try:
        file_bytes = io.BytesIO(uploaded_file.read())
        npz = np.load(file_bytes, allow_pickle=True)
    except Exception as exc:
        st.error(f"Failed to read .npz: {exc}")
        return

    try:
        X_pad, meta = render_sequence_overview(npz, cfg["sequence_length"])
    except Exception as exc:
        st.error(str(exc))
        return

    render_feature_charts(X_pad)

    st.subheader("Predictions (simulated)")
    rng = np.random.RandomState(cfg["random_seed"])  # reproducible
    gloss_logits, cat_logits = simulate_predictions(
        rng, cfg["num_gloss_classes"], cfg["num_category_classes"]
    )

    g_idx, g_prob = topk_from_logits(gloss_logits, cfg["topk"]) 
    c_idx, c_prob = topk_from_logits(cat_logits, cfg["topk"]) 

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Top-K Gloss**")
        render_topk_table(g_idx, g_prob, label_prefix="gloss")
    with cols[1]:
        st.markdown("**Top-K Category**")
        render_topk_table(c_idx, c_prob, label_prefix="cat")


if __name__ == "__main__":
    main()


