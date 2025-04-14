from .constants import DEBUG

def logger(data, title=None, prefix=""):
    if not DEBUG:
        return

    if isinstance(data, (float, int)):
        print(f"[DEBUG] {prefix}{title}: {data:.4f}")

    elif isinstance(data, dict):
        print(f"[DEBUG] {title or 'Metrics'}:")
        for k, v in data.items():
            print(f"  {prefix}{k}: {v:.4f}")

    elif isinstance(data, tuple) and len(data) == 3:
        inputs, refs, preds = data
        n = min(5, len(inputs))
        print(f"\n[DEBUG] {title or 'Sample Predictions'} — Showing {n} examples:")
        for i in range(n):
            print(f"Input    : {inputs[i]}")
            print(f"Expected : {refs[i]}")
            print(f"Predicted: {preds[i]}\n")
