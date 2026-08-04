import argparse
import os

# dont scale this networks since they are already very small and scaling causes more overhead than benefits
triton_scaling_bypass = {
    "wc",
    "doors",
    "ppe",
    "rtmpose",
    "hands",
    "phone",
    "stgcn",
    "expert-detect",
    "violence-detect",
    "violence",
    "image-quality",
    "lpc",
    "container-ocd",
}

def add_instance_group_to_triton_configs(model_repository: str = None, num_instances: int = 1):
    """
    Appends instance_group to all config.pbtxt files in the Triton model_repository.
    Assumes no instance_group exists in the config files.
    """
    if model_repository is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_repository = os.path.join(script_dir, "model_repository")
    instance_group_str = (
        "instance_group [\n" "  {\n" f"    count: {num_instances}\n" "    kind: KIND_GPU\n" "  }\n" "]\n"
    )
    for model_name in os.listdir(model_repository):
        if model_name in triton_scaling_bypass:
            continue
        model_dir = os.path.join(model_repository, model_name)
        config_path = os.path.join(model_dir, "config.pbtxt")
        if os.path.isfile(config_path):
            with open(config_path, "a") as f:
                f.write("\n" + instance_group_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add instance_group to Triton model configs.")
    parser.add_argument("-m", "--model_repository", type=str, default=None, help="Path to the Triton model repository.")
    parser.add_argument(
        "-n", "--num_instances", type=int, default=1, help="Number of instances to set in instance_group."
    )

    args = parser.parse_args()
    if args.num_instances > 1:
        add_instance_group_to_triton_configs(args.model_repository, args.num_instances)
        print(
            f"Appended instance_group to all config.pbtxt files in {args.model_repository or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_repository')} with {args.num_instances} instances."
        )
    else:
        print("Number of instances must be greater than 1. No changes made.")
