# tools/get_latest_run.py
import json, os, sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python tools/get_latest_run.py <EXP_DIR>", file=sys.stderr)
        sys.exit(2)

    exp_dir = sys.argv[1]
    marker = os.path.join(exp_dir, "latest_run.txt")
    if not os.path.isfile(marker):
        print(f"latest_run.txt not found at: {marker}", file=sys.stderr)
        sys.exit(1)

    with open(marker, "r") as f:
        content = f.read().strip()

    # Support both plain path (run_dir) and JSON payload
    run_dir, results_dir = "", ""
    try:
        data = json.loads(content)
        run_dir = data.get("run_dir") or ""
        results_dir = data.get("results_folder") or (os.path.join(run_dir, "track_results") if run_dir else "")
    except json.JSONDecodeError:
        run_dir = content
        results_dir = os.path.join(run_dir, "track_results")

    if not results_dir or not os.path.isdir(results_dir):
        print(f"Resolved results_dir does not exist: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(results_dir)  # <- stdout only the path so bash can capture

if __name__ == "__main__":
    main()