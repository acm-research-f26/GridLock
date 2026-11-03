# Unconstrained targetted attack

import argparse
import json
from pathlib import Path
import numpy as np

from GridData import ML_OUTPUT, prepare_rows, sequence_windows


def build_ieee39_lstm_test(
    csv_path,
    offsets_raw,
    start_timestep,
    stop_timestep,
    window=24,
):
    #Build paired clean/corrupted TEST windows.

    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer.")

    for name, value in (
        ("start_timestep", start_timestep),
        ("stop_timestep", stop_timestep),
    ):
        if not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer.")

    if start_timestep >= stop_timestep:
        raise ValueError("Require start_timestep < stop_timestep.")

    if not isinstance(offsets_raw, dict) or not offsets_raw:
        raise ValueError("offsets_raw must be a nonempty dictionary.")

    frame, clean_values, metadata = prepare_rows(csv_path)

    features = list(metadata["feature_columns"])
    lookup = {name: i for i, name in enumerate(features)}

    graph = metadata.get("graph")
    if graph is not None and graph.get("node_count") != 39:
        raise ValueError("Graph metadata does not describe 39 buses.")

    unknown = sorted(set(offsets_raw) - set(features))
    if unknown:
        raise ValueError(f"Unknown dataset features: {unknown}")

    raw_offsets = {
        name: float(value)
        for name, value in offsets_raw.items()
    }

    if not all(np.isfinite(value) for value in raw_offsets.values()):
        raise ValueError("Offsets must be finite.")

    if not any(value != 0.0 for value in raw_offsets.values()):
        raise ValueError("At least one offset must be nonzero.")

    scale = np.asarray(metadata["scale"], dtype=np.float64)

    if scale.shape != (len(features),):
        raise ValueError("Normalization scale does not match feature count.")

    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Normalization scales must be finite and positive.")

    steps = frame["timestep"].to_numpy(dtype=np.int64)
    splits = frame["split"].to_numpy()

    interval_rows = (
        (steps >= start_timestep)
        & (steps < stop_timestep)
    )

    if np.any(interval_rows & (splits != "test")):
        raise ValueError(
            "The interval includes retained training or validation rows. "
            "Choose a TEST-only interval."
        )

    affected_rows = interval_rows & (splits == "test")

    if not affected_rows.any():
        raise ValueError("No accepted TEST rows occur in this interval.")

    corrupted_values = clean_values.copy()
    normalized_offsets = {}

    for name, offset_raw in raw_offsets.items():
        column = lookup[name]

        # For z = (x - mean) / scale, an additive raw-unit offset
        # becomes offset_raw / scale in normalized coordinates.
        offset_normalized = offset_raw / scale[column]

        corrupted_values[affected_rows, column] += offset_normalized
        normalized_offsets[name] = float(offset_normalized)

    if not np.isfinite(corrupted_values).all():
        raise ValueError("Corruption produced nonfinite measurements.")

    with np.errstate(over="ignore", invalid="ignore"):
        clean_finite = np.isfinite(
            clean_values.astype(np.float32)
        ).all()
        corrupted_finite = np.isfinite(
            corrupted_values.astype(np.float32)
        ).all()

    if not clean_finite or not corrupted_finite:
        raise ValueError("Measurements overflow float32 model inputs.")

    clean_windows, clean_ends = sequence_windows(
        clean_values, steps, splits, window
    )

    corrupted_windows, corrupted_ends = sequence_windows(
        corrupted_values, steps, splits, window
    )

    label_windows, label_ends = sequence_windows(
        affected_rows.astype(np.float32)[:, None],
        steps,
        splits,
        window,
    )

    X_clean = clean_windows["test"]
    X_corrupted = corrupted_windows["test"]
    end_timesteps = clean_ends["test"]

    if len(X_clean) == 0:
        raise ValueError("No contiguous TEST windows are available.")

    if not (
        np.array_equal(end_timesteps, corrupted_ends["test"])
        and np.array_equal(end_timesteps, label_ends["test"])
    ):
        raise RuntimeError("Clean, corrupted, and label windows misalign.")

    timestep_labels = label_windows["test"][:, :, 0].astype(bool)
    window_labels = timestep_labels.any(axis=1)
    endpoint_labels = timestep_labels[:, -1]

    if not window_labels.any():
        raise ValueError(
            "Affected rows are not included in any valid TEST window."
        )

    window_timesteps = (
        end_timesteps[:, None]
        - np.arange(window - 1, -1, -1)[None, :]
    )

    report = {
        "test_type": "bounded_multi_feature_telemetry_corruption",
        "source_csv": str(csv_path),
        "window_length": int(window),
        "feature_count": len(features),
        "test_window_count": int(len(X_clean)),
        "affected_window_count": int(window_labels.sum()),
        "affected_endpoint_count": int(endpoint_labels.sum()),
        "affected_retained_row_count": int(affected_rows.sum()),
        "start_timestep_inclusive": int(start_timestep),
        "stop_timestep_exclusive": int(stop_timestep),
        "affected_timesteps": steps[affected_rows].tolist(),
        "offsets_original_units": raw_offsets,
        "offsets_normalized_units": normalized_offsets,
        "normalization_version": metadata["normalization_version"],
        "label_policy": (
            "Physical ground truth remains unchanged. Separate labels "
            "identify corrupted input rows, windows, and endpoints."
        ),
        "consistency_policy": (
            "Only named telemetry columns are modified. Related derived "
            "measurements are not recomputed."
        ),
    }

    return {
        "X_clean": X_clean,
        "X_corrupted": X_corrupted,
        "window_labels": window_labels,
        "endpoint_labels": endpoint_labels,
        "timestep_labels": timestep_labels,
        "window_timesteps": window_timesteps,
        "end_timesteps": end_timesteps,
        "feature_names": features,
        "report": report,
    }


def compare_keras_outputs(
    model,
    bundle,
    batch_size=64,
    model_feature_names=None,
):
    """
    Run paired inputs through an already-loaded Keras model.

    Supply model_feature_names if training selected or reordered features.
    Supports a single array output; no detection threshold is inferred.
    """
    X_clean = bundle["X_clean"]
    X_corrupted = bundle["X_corrupted"]

    if model_feature_names is not None:
        if not model_feature_names:
            raise ValueError("model_feature_names cannot be empty.")

        if len(model_feature_names) != len(set(model_feature_names)):
            raise ValueError("Model feature names must be unique.")

        lookup = {
            name: i
            for i, name in enumerate(bundle["feature_names"])
        }

        unknown = sorted(set(model_feature_names) - set(lookup))
        if unknown:
            raise ValueError(f"Unknown model features: {unknown}")

        positions = [lookup[name] for name in model_feature_names]
        X_clean = X_clean[:, :, positions]
        X_corrupted = X_corrupted[:, :, positions]

    clean_output = model.predict(
        X_clean, batch_size=batch_size, verbose=0
    )
    corrupted_output = model.predict(
        X_corrupted, batch_size=batch_size, verbose=0
    )

    if isinstance(clean_output, (dict, list, tuple)) or isinstance(
        corrupted_output, (dict, list, tuple)
    ):
        raise ValueError(
            "This adapter expects one array output. Select the desired "
            "output explicitly for a multi-output model."
        )

    clean_output = np.asarray(clean_output)
    corrupted_output = np.asarray(corrupted_output)

    if clean_output.shape != corrupted_output.shape:
        raise ValueError("Clean and corrupted output shapes differ.")

    if (
        clean_output.ndim == 0
        or clean_output.shape[0] != len(X_clean)
    ):
        raise ValueError("Model outputs must retain the sample dimension.")

    if not (
        np.isfinite(clean_output).all()
        and np.isfinite(corrupted_output).all()
    ):
        raise ValueError("Model predictions contain nonfinite values.")

    return {
        "clean_output": clean_output,
        "corrupted_output": corrupted_output,
        "output_difference": corrupted_output - clean_output,
    }


def save_test_bundle(bundle, output_path, predictions=None):
    """Save paired inputs, labels, optional predictions, and metadata."""
    output_path = Path(output_path)

    if output_path.suffix.lower() != ".npz":
        raise ValueError("Output path must end in .npz.")

    report_path = Path(str(output_path) + ".json")

    if output_path.exists() or report_path.exists():
        raise FileExistsError(
            "Output already exists. Choose a new output path."
        )

    arrays = {
        "X_clean": bundle["X_clean"],
        "X_corrupted": bundle["X_corrupted"],
        "window_labels": bundle["window_labels"],
        "endpoint_labels": bundle["endpoint_labels"],
        "timestep_labels": bundle["timestep_labels"],
        "window_timesteps": bundle["window_timesteps"],
        "end_timesteps": bundle["end_timesteps"],
        "feature_names": np.asarray(bundle["feature_names"], dtype=str),
    }

    if predictions is not None:
        arrays.update(predictions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)

    report_path.write_text(
        json.dumps(bundle["report"], indent=2, allow_nan=False) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--csv", default=ML_OUTPUT)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)

    parser.add_argument(
        "--offsets",
        default=json.dumps({
            "Bus16_vm_pu": 0.02,
            "Bus17_vm_pu": -0.01,
            "Bus18_vm_pu": 0.01,
        }),
        help="JSON object mapping feature names to raw-unit offsets.",
    )

    parser.add_argument(
        "--output",
        default="Dataset/ieee39_lstm_corruption_test.npz",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Optional path to a saved Keras model.",
    )

    parser.add_argument(
        "--model-features",
        default=None,
        help=(
            "Optional JSON file containing the exact ordered list of "
            "features used by the trained LSTM."
        ),
    )

    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    try:
        offsets = json.loads(args.offsets)
    except json.JSONDecodeError as exc:
        parser.error(f"Invalid --offsets JSON: {exc}")

    if not isinstance(offsets, dict):
        parser.error("--offsets must be a JSON object.")

    bundle = build_ieee39_lstm_test(
        csv_path=args.csv,
        offsets_raw=offsets,
        start_timestep=args.start,
        stop_timestep=args.stop,
        window=args.window,
    )

    predictions = None

    if args.model is not None:
        from tensorflow.keras.models import load_model

        model_features = None

        if args.model_features is not None:
            model_features = json.loads(
                Path(args.model_features).read_text()
            )

            if not isinstance(model_features, list) or not all(
                isinstance(name, str) for name in model_features
            ):
                raise ValueError(
                    "Model feature file must contain a JSON list of names."
                )

        # Custom layers may require your project's custom_objects mapping.
        model = load_model(args.model, compile=False)

        predictions = compare_keras_outputs(
            model=model,
            bundle=bundle,
            batch_size=args.batch_size,
            model_feature_names=model_features,
        )

        bundle["report"]["model_path"] = str(args.model)
        bundle["report"]["model_feature_names"] = (
            model_features
            if model_features is not None
            else bundle["feature_names"]
        )

    elif args.model_features is not None:
        parser.error("--model-features requires --model.")

    save_test_bundle(
        bundle=bundle,
        output_path=args.output,
        predictions=predictions,
    )

    print(json.dumps(bundle["report"], indent=2))
    print("Clean input shape:", bundle["X_clean"].shape)
    print("Corrupted input shape:", bundle["X_corrupted"].shape)

    if predictions is not None:
        print("Model output shape:", predictions["clean_output"].shape)

    print("Saved:", args.output)
    print(
        "Apply your existing scoring method and frozen threshold to "
        "evaluate detection; output changes alone are not detections."
    )


if __name__ == "__main__":
    main()