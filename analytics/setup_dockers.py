import argparse
import concurrent.futures
import glob
import os
import shutil
import subprocess
from pathlib import Path

from analyzer_manager.cuda_library.build_cuda import print_subprocess_results, build_all
from analyzer_manager.tools.trt_exporter.export_common import (
    is_x86_cuda_platform,
    is_jetson_platform,
    HostType,
    determine_host,
)

BASE_DOCKER_NAME = "lumixai/analytic_manager"
TRITON_BASE_NAME = "lumixai/inference_server"
OFFLINE_BASE_NAME = "lumixai/offline_analytics"
EXPORTER_BASE_NAME = "lumixai/pf_exporter"
LOCAL_VCC_BASE_NAME = "lumixai/local_vcc"

WEIGHTS_BUCKET = "lumix-analytics-default-weights"
LOCAL_VCC_MODEL_PREFIX = "cloud/oracle/Qwen3_8B/"
LOCAL_VCC_MODELS_DIR = os.path.join("analyzer_manager", "local_vcc", "models")
LOCAL_VCC_VERSION = "0.0.3"

default_env = {
    "APP_ENV": "prod",
    "VIDEO_TEST": True,
    "BURST_TEST": False,
    "JETSON": False,
    "CONSOLE_LOG": False,
    "TRITON_FORCED": True,
}

# weights that are currently not supported by host
weights_blacklist = {
    "motion-bert": set(HostType),
    "yolov5": set(HostType),
}

no_triton_hosts = [HostType.NONE, HostType.XAVIER]
blackwell_hosts = [HostType.RTX2000PRO, HostType.RTX4000PRO, HostType.RTX4070]
triton_instance_scaling = {HostType.L4: 2, HostType.RTX4000PRO: 2}


def get_latest_s3_file(bucket, prefix, local_destination):
    """
    Finds the latest file (by timestamp) in an S3 bucket with a given prefix and downloads it.

    Args:
        bucket (str): S3 bucket name.
        prefix (str): Prefix to filter files.
        local_destination (str): Local path to save the downloaded file.

    Returns:
        str: The downloaded file path or None if no files were found.
    """
    try:
        # Step 1: List files in S3 under the given prefix
        list_command = f"aws s3 ls s3://{bucket}/{prefix} --recursive"
        result = subprocess.run(list_command, shell=True, capture_output=True, text=True)

        if result.returncode != 0 or not result.stdout.strip():
            print(f"No files found with the given prefix {prefix}")
            return None

        # Step 2: Parse the file list and extract filenames
        files = [line.split()[-1] for line in result.stdout.strip().split("\n")]

        if not files:
            print(f"No valid files found for {prefix}")
            return None

        if len(files) > 1:
            print(f"Multiple files found for {prefix}, can't select the appropriate one automatically")
            return None

        # Step 3: Select the most recent file (last in sorted order)
        latest_file = sorted(files)[-1]
        if os.path.exists(os.path.join(local_destination, latest_file)):
            return latest_file

        # Step 4: Download the selected file
        os.makedirs(local_destination, exist_ok=True)
        download_command = f"aws s3 cp s3://{bucket}/{latest_file} {local_destination} --no-progress"
        subprocess.run(download_command, shell=True, check=True)
        print(f"Downloaded: {latest_file} to {local_destination}")
        return latest_file

    except Exception as e:
        print(f"Error: {e}")
        return None


def read_repo_version():
    with open("version", "rt") as f:
        vers = f.readline()
    return vers.strip(" ").strip("\n")


def parse_commandline() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--compile-cuda", action="store_true", help="compile cuda code")
    parser.add_argument(
        "-f",
        "--force-host",
        type=HostType,
        default=HostType.NONE,
        choices=[v.value for v in HostType],
        help="force to build for a specific host",
    )
    parser.add_argument("-v", "--tag-version", type=str, default="", help="tag for base docker building")
    parser.add_argument("-b", "--build", action="store_true", help="for building base docker")
    parser.add_argument("-d", "--dev", action="store_true", help="for development environment")
    parser.add_argument("-t", "--triton", action="store_true", help="generate triton docker")
    parser.add_argument("-o", "--offline", action="store_true", help="generate offline analytics docker")
    parser.add_argument("-sd", "--skip-download", action="store_true", help="skips s3 sync")
    parser.add_argument("-e", "--export", action="store_true", help="generate exporter docker")
    parser.add_argument("-re", "--reset-export", action="store_true", help="rebuild exporter base")
    parser.add_argument(
        "-lv",
        "--local-vcc",
        action="store_true",
        help="sync the local_vcc VLM model from s3 and build the local_vcc docker (tagged lumixai/local_vcc:latest)",
    )
    parser.add_argument(
        "-p",
        "--push",
        action="store_true",
        help="with --local-vcc: also tag the built image as LOCAL_VCC_VERSION and push it to docker hub",
    )

    parsed = parser.parse_args()
    if parsed.reset_export:
        parsed.export = True

    return parsed


def handle_cuda_compilation(force_compile: bool, host: HostType):
    host_support = is_x86_cuda_platform(host) or is_jetson_platform(host)
    cuda_folder = r"analyzer_manager/cuda_library"
    files_to_compile = [
        "parse_images_cuda",
        "image_extractor",
        "motion_estimator",
        "motion_estimation_extractor",
    ]

    existing_cuda_files = glob.glob(os.path.join(cuda_folder, "*.so"))
    if host_support and (len(existing_cuda_files) < len(files_to_compile) and force_compile):
        build_all(nvcc_path="nvcc")

    # nvjpeg encoder handling
    nvjpeg_encoder = "nvJpeg_encoder.so"
    if is_jetson_platform(host) and not os.path.exists(os.path.join(cuda_folder, nvjpeg_encoder)) and force_compile:
        if host == HostType.XAVIER:
            build_script = "build.sh"
        else:
            build_script = "build_orin.sh"
        build_res = subprocess.run(["bash", build_script], cwd=cuda_folder)
        print_subprocess_results(build_res)


def generate_tags(repo_tag: str, host: HostType, args) -> (str, str):
    # write base docker
    tag_version = f":{read_repo_version()}"
    extension = ""
    if args.tag_version:
        extension += "-" + args.tag_version
    if host is not HostType.XAVIER:
        extension += "-" + host.value
    base_tag = repo_tag + tag_version + extension
    analyzer_tag = base_tag
    # if not args.build_base:
    #    analyzer_tag = f"{repo_tag}:{DEFAULT_BASE_ANALYZER_VERSION_TO_USE}{extension}"
    return base_tag, analyzer_tag


def copy_config_specific_weights_inference_server(host: HostType, is_dev=False, skip_download=False):
    weights_out_path = os.path.join("analyzer_manager", "inference", "model_repository")
    weights_in_path = os.path.join("weights", host.value)
    all_weights_path = os.path.join("weights", "all")
    if not skip_download:
        print(f"syncing weights with s3 --> {weights_in_path}")
        models_to_sync = {}
        for model_name in os.listdir(all_weights_path):
            if weights_blacklist.get(model_name, set()) & {host}:  # skip blacklisted models
                continue
            files = os.listdir(os.path.join(all_weights_path, model_name))
            if len(files) > 0:
                models_to_sync[model_name] = files[-1]

        models = list(models_to_sync.keys())

        def sync_task(idx):
            cur_model_name = models[idx]
            cur_file = models_to_sync[cur_model_name]
            out_dir = os.path.join(weights_in_path, cur_model_name)
            prefix = Path(cur_file).stem
            existing = glob.glob(os.path.join(out_dir, prefix + "*"))
            if existing:
                return True
            else:
                fname = get_latest_s3_file(WEIGHTS_BUCKET, "/".join([host.value, cur_model_name, prefix]), out_dir)
                return fname is not None

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(sync_task, range(len(models))))
        if not all(results) and not is_dev:
            raise FileNotFoundError("failed to download required weights")

    ext_name_map = {"engine": "plan"}
    version = "1"

    def copy_weight_to_server(in_path: str, is_rename: bool = True):
        for curr_model_name in os.listdir(in_path):
            curr_files = os.listdir(os.path.join(in_path, curr_model_name))
            for curr_file in curr_files:
                in_file = os.path.join(in_path, curr_model_name, curr_file)
                if is_rename:
                    _, ext_ = os.path.splitext(curr_file)
                    if ext_[1:] in ext_name_map:
                        target_file = "model." + ext_name_map[ext_[1:]]
                    else:
                        target_file = "model" + ext_
                else:
                    target_file = curr_file
                out_dir = os.path.join(weights_out_path, curr_model_name, version)
                os.makedirs(out_dir, exist_ok=True)
                shutil.copy2(in_file, os.path.join(out_dir, target_file))

    # clean old files
    for ext in ["engine", "plan", "pt", "onnx"]:
        platform_files = glob.glob(os.path.join(weights_out_path, "**", "**", "*." + ext))
        for file in platform_files:
            print(f"found platform file {file}.... removing")
            os.remove(file)

    if is_dev:
        weights_in_all_path = os.path.join("weights", "all")
        print(f"copying shared files from  {weights_in_all_path}")
        copy_weight_to_server(weights_in_all_path, is_rename=False)

    # make sure path exists
    os.makedirs(weights_in_path, exist_ok=True)
    print(f"copying shared files from  {weights_in_path}")
    copy_weight_to_server(weights_in_path, is_rename=True)


def copy_config_specific_weights(host: HostType):
    if not is_jetson_platform(host):
        return

    weights_in_path = os.path.join("weights", host.value)
    weights_out_path = os.path.join("analyzer_manager", "resources", "weights")

    # clean old files
    platform_files = glob.glob(os.path.join(weights_out_path, "**", "*.engine"))
    for file in platform_files:
        print(f"found platform file {file}.... removing")
        os.remove(file)

    def copytree(src, dst, symlinks=False, ignore=None):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)

    print(f"copying platform files from  {weights_in_path}")
    copytree(weights_in_path, weights_out_path)


def generate_triton_docker(my_host: HostType):
    if my_host is HostType.ORIN_JP5:
        base_docker = "lumixai/triton_server:0.0.2-orin"
    elif my_host is HostType.ORIN_JP6:
        base_docker = "lumixai/triton_server:0.0.3-" + my_host
    elif my_host is HostType.XAVIER:
        base_docker = "python:3.6-slim-buster"  # placeholder - doesn't actually work
    elif my_host in blackwell_hosts:
        base_docker = "nvcr.io/nvidia/tritonserver:25.05-py3"
    else:
        base_docker = "nvcr.io/nvidia/tritonserver:24.01-py3"

    triton_docker_name = "Dockerfile"
    docker_out_path = os.path.join("analyzer_manager", "inference", triton_docker_name)
    docker_in_path = os.path.join("analyzer_manager", "inference", "DockerfileTriton.common")

    # Open the text file in read mode and read its content
    with open(docker_in_path, "r") as f:
        common_content = f.read()

    scale_factor = triton_instance_scaling.get(my_host, 1)
    scaling_env = f"ENV TRITON_INSTANCE_SCALING={scale_factor}\n"

    # Open a new file in write mode, write some text to it, then append the text from the existing file
    with open(docker_out_path, "w") as f:
        f.write("# Autogenerated with setup_dockers.py\n")
        f.write(f"FROM {base_docker}\n")
        f.write(scaling_env)
        f.write(common_content)
    return docker_out_path


def run_full_exporter(host: HostType, base_tag: str = None):
    common_docker = analyzer_path("DockerfileExporter.common")
    out_docker = common_docker.split(".")[0]

    # generate exporter docker file
    with open(out_docker, "wt") as outfile:
        outfile.write("# Autogenerated with setup_dockers.py\n")
        outfile.write(f"FROM {base_tag}\n\n")

        with open(common_docker, "rt") as infile:
            for line in infile:
                outfile.write(line)

    # build exporter docker
    docker_name = "exporter"
    out_docker = Path(out_docker).name
    print(f"building exporter docker {out_docker}")
    my_env = os.environ.copy()
    p = subprocess.run(
        args=["docker", "build", "-t", docker_name, "-f", out_docker, "."],
        cwd=analyzer_path(""),
        env=my_env,
    )
    print_subprocess_results(p)

    weights_out_path = os.path.join(os.getcwd(), "weights", host)
    os.makedirs(weights_out_path, exist_ok=True)
    print("running exporter docker")
    p = subprocess.run(
        args=[
            "docker",
            "run",
            "--runtime",
            "nvidia",
            "-v",
            f"{weights_out_path}:/exports",
            "-v",
            f"{os.path.join(os.getcwd(), 'weights/all')}:/model_repository",
            docker_name,
            "./build_all.sh",
        ],
        env=my_env,
    )
    print_subprocess_results(p)


def analyzer_path(file_path):
    return os.path.join("analyzer_manager", file_path)


def sync_local_vcc_model(skip_download: bool = False):
    """Download the local_vcc VLM (model + mmproj) from s3 into
    ``analyzer_manager/local_vcc/models/`` as ``model.gguf`` / ``mmproj.gguf``
    (the names ``Dockerfile.local_vcc`` expects)."""
    model_path = os.path.join(LOCAL_VCC_MODELS_DIR, "model.gguf")
    mmproj_path = os.path.join(LOCAL_VCC_MODELS_DIR, "mmproj.gguf")

    if skip_download:
        print("skipping local_vcc model sync (--skip-download)")
        return

    if os.path.exists(model_path) and os.path.exists(mmproj_path):
        print(f"local_vcc model files already present under {LOCAL_VCC_MODELS_DIR}, skipping sync")
        return

    print(f"syncing local_vcc model from s3://{WEIGHTS_BUCKET}/{LOCAL_VCC_MODEL_PREFIX}")
    list_command = f"aws s3 ls s3://{WEIGHTS_BUCKET}/{LOCAL_VCC_MODEL_PREFIX} --recursive"
    result = subprocess.run(list_command, shell=True, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise FileNotFoundError(f"no files found under s3://{WEIGHTS_BUCKET}/{LOCAL_VCC_MODEL_PREFIX}")

    keys = [line.split()[-1] for line in result.stdout.strip().split("\n")]
    gguf_keys = [k for k in keys if k.lower().endswith(".gguf")]
    mmproj_keys = [k for k in gguf_keys if "mmproj" in k.lower()]
    model_keys = [k for k in gguf_keys if "mmproj" not in k.lower()]

    if len(mmproj_keys) != 1 or len(model_keys) != 1:
        raise FileNotFoundError(
            f"expected exactly one *mmproj*.gguf and one model .gguf under "
            f"s3://{WEIGHTS_BUCKET}/{LOCAL_VCC_MODEL_PREFIX}, found: {gguf_keys}"
        )

    os.makedirs(LOCAL_VCC_MODELS_DIR, exist_ok=True)
    for key, dest, label in [
        (model_keys[0], model_path, "model"),
        (mmproj_keys[0], mmproj_path, "mmproj"),
    ]:
        print(f"downloading {label} gguf: s3://{WEIGHTS_BUCKET}/{key} -> {dest}")
        subprocess.run(
            ["aws", "s3", "cp", f"s3://{WEIGHTS_BUCKET}/{key}", dest, "--no-progress"],
            check=True,
        )


def build_local_vcc_docker(skip_download: bool = False, push: bool = False):
    sync_local_vcc_model(skip_download)

    latest_tag = f"{LOCAL_VCC_BASE_NAME}:latest"
    print(f"building local_vcc docker {latest_tag}")
    my_env = os.environ.copy()
    p = subprocess.run(
        args=["docker", "build", "-f", "Dockerfile.local_vcc", "-t", latest_tag, "."],
        cwd=analyzer_path(""),
        env=my_env,
    )
    print_subprocess_results(p)

    if push:
        version_tag = f"{LOCAL_VCC_BASE_NAME}:{LOCAL_VCC_VERSION}"
        print(f"tagging {latest_tag} as {version_tag}")
        p = subprocess.run(args=["docker", "tag", latest_tag, version_tag])
        print_subprocess_results(p)

        print(f"pushing {version_tag} to docker hub")
        p = subprocess.run(args=["docker", "push", version_tag])
        print_subprocess_results(p)

        print(f"pushing {version_tag} to docker hub")
        p = subprocess.run(args=["docker", "push", latest_tag])
        print_subprocess_results(p)


def setup_dockers():
    args = parse_commandline()

    if args.local_vcc:
        build_local_vcc_docker(skip_download=args.skip_download, push=args.push)
        print("Done")
        return

    my_host = determine_host() if args.force_host == HostType.NONE else args.force_host
    # handle_cuda_compilation(args.compile_cuda, my_host)

    # helper function to wrap path stuff

    # per host configuration
    env_data = default_env
    base_docker_file = "Dockerfile"
    docker_file = base_docker_file + "." + my_host
    env_data["HOST"] = my_host.value
    if is_jetson_platform(my_host):
        env_data["JETSON"] = True
    elif is_x86_cuda_platform(my_host):
        env_data["BURST_TEST"] = True
        if my_host in blackwell_hosts:
            docker_file = base_docker_file + ".blackwell"
        else:
            docker_file = base_docker_file + ".cpu"
    elif my_host == HostType.CPU:
        pass
    else:
        raise ValueError("cant resolve host")

    # handle required files from other repose, etc.
    # copy config file
    print("copying config file")
    destination_file = os.path.join(Path.home(), "configs", "config.json")

    try:
        os.remove(destination_file)
    except FileNotFoundError:
        pass
    shutil.copyfile(os.path.join("analyzer_manager", "assets", "config.json"), destination_file)

    destination_file = os.path.join(Path.home(), "configs", "configAnalytic.json")

    try:
        os.remove(destination_file)
    except FileNotFoundError:
        pass
    shutil.copyfile(
        os.path.join("analyzer_manager", "assets", "configAnalytic.json"),
        destination_file,
    )

    # if we are in an environment where analytics initialized analyzer_manager/automation,
    #   copy docker_start.sh from analyzer_manager/automation.
    #   otherwise, we assume that this script was running from tsc_all.sh,
    #   and in that case docker_start.sh should have already been copied
    source_file = os.path.join(
        "analyzer_manager", "automation", "scripts", "edge_installation", "infra", "docker_start.sh"
    )
    destination_file = os.path.join("analyzer_manager", "docker_start.sh")
    if os.path.exists(source_file):
        print("copying docker_start.sh")

        try:
            os.remove(destination_file)
        except FileNotFoundError:
            pass
        shutil.copyfile(source_file, destination_file)

    if not os.path.exists(destination_file):
        raise FileNotFoundError("docker_start.sh not found")

    if args.triton or args.build or args.export:
        copy_config_specific_weights_inference_server(
            my_host, args.dev or args.export, args.skip_download or args.export
        )

    # generate .env file
    with open(".env", "wt") as outfile:
        for key in env_data.keys():
            value = f"'{env_data[key]}'" if type(env_data[key]) is bool else env_data[key]
            outfile.write(f"{key} = {value}\n")

    base_tag, analyzer_tag = generate_tags(BASE_DOCKER_NAME, my_host, args)
    print(f"generating analyzer manager base docker {base_tag}")
    out_docker_file = "Dockerfile"
    dockers_to_concatenate = [docker_file, "DockerfileBase.common", "Dockerfile.common"]
    with open(analyzer_path(out_docker_file), "wt") as outfile:
        outfile.write("# Autogenerated with setup_dockers.py\n")
        for dock_file in dockers_to_concatenate:
            with open(analyzer_path(dock_file), "rt") as infile:
                for line in infile:
                    outfile.write(line)

    # set the docker ignore to include the large folders in the repo
    out_docker_ignore_file = out_docker_file + ".dockerignore"
    exclude_from_ignore = ["resources", "inference", "cuda_library", "python_library"]
    with open(analyzer_path(out_docker_ignore_file), "wt") as outfile:
        outfile.write("# Autogenerated with setup_dockers.py\n")
        with open(analyzer_path(".dockerignore"), "rt") as infile:
            for line in infile:
                if line.strip("\n") not in exclude_from_ignore:
                    outfile.write(line)

    # with open(analyzer_path("Dockerfile"), "wt") as outfile:
    #    outfile.write("# Autogenerated with setup_dockers.py\n")
    #    outfile.write("# build using correct base image\n")
    #    outfile.write(f"FROM {analyzer_tag}\n")

    #    with open(analyzer_path("Dockerfile.common"), "rt") as infile:
    #        for line in infile:
    #            outfile.write(line)

    # handle tester docker
    print(f"generating analyzer tester docker from  {docker_file}")
    tester_in_docker = os.path.join("analyzer_tester", docker_file)
    tester_out_docker = os.path.join("analyzer_tester", "Dockerfile")
    try:
        os.remove(tester_out_docker)
    except FileNotFoundError:
        pass
    shutil.copyfile(tester_in_docker, tester_out_docker)

    # build if necessary
    if args.build or args.reset_export:
        print(f"building analytic docker {base_tag}")
        my_env = os.environ.copy()
        my_env["DOCKER_BUILDKIT"] = "1"
        p = subprocess.run(
            args=["docker", "build", "-t", base_tag, "-f", out_docker_file, "."],
            cwd=analyzer_path(""),
            env=my_env,
        )
        print_subprocess_results(p)

        print(f"tagging {base_tag} as latest")
        p = subprocess.run(
            args=["docker", "tag", base_tag, f"{BASE_DOCKER_NAME}:latest"],
            cwd=analyzer_path(""),
        )
        print_subprocess_results(p)

    if args.triton and my_host not in no_triton_hosts:
        _, triton_tag = generate_tags(TRITON_BASE_NAME, my_host, args)
        triton_docker_file = generate_triton_docker(my_host)
        print(f"triton docker file {triton_docker_file} generating with tag {triton_tag}")

        if args.build:
            print(f"building triton docker {triton_tag}")
            my_env = os.environ.copy()
            p = subprocess.run(
                args=["docker", "build", "-t", triton_tag, "."],
                cwd=analyzer_path("inference"),
                env=my_env,
            )
            print_subprocess_results(p)

            print(f"tagging {triton_tag} as latest")
            p = subprocess.run(
                args=["docker", "tag", triton_tag, f"{TRITON_BASE_NAME}:latest"],
                cwd=analyzer_path(""),
            )
            print_subprocess_results(p)

    exporter_tag = f"{EXPORTER_BASE_NAME}:latest-{my_host.value}"
    if args.reset_export:
        my_env = os.environ.copy()
        _, exporter_base_tag = generate_tags(EXPORTER_BASE_NAME, my_host, args)
        print(f"rebuilding exporter base docker {exporter_base_tag}")
        docker_file = "DockerfileExporterBase"
        p = subprocess.run(
            args=["docker", "build", "-f", docker_file, "-t", exporter_base_tag, "."],
            cwd=analyzer_path(""),
            env=my_env,
        )
        print_subprocess_results(p)

        print(f"tagging {exporter_base_tag} as latest")
        p = subprocess.run(
            args=["docker", "tag", exporter_base_tag, exporter_tag],
            cwd=analyzer_path(""),
        )
        print_subprocess_results(p)

    if args.export:
        print("generating exporter docker")

        out_docker_file = "DockerfileExporter"
        dockers_to_concatenate = ["DockerfileExporter.common"]
        with open(analyzer_path(out_docker_file), "wt") as outfile:
            outfile.write("# Autogenerated with setup_dockers.py\n")
            outfile.write(f"FROM {exporter_tag}\n\n")
            for dock_file in dockers_to_concatenate:
                with open(analyzer_path(dock_file), "rt") as infile:
                    for line in infile:
                        outfile.write(line)

        my_env = os.environ.copy()
        p = subprocess.run(
            args=["docker", "build", "-f", out_docker_file, "-t", "exporter", "."],
            cwd=analyzer_path(""),
            env=my_env,
        )
        print_subprocess_results(p)

    if args.offline:
        _, offline_tag = generate_tags(OFFLINE_BASE_NAME, my_host, args)
        if args.build:
            print(f"building offline docker {offline_tag}")
            my_env = os.environ.copy()
            p = subprocess.run(
                args=["docker", "build", "-t", offline_tag, "-f", "DockerfileOffline", "."],
                cwd=analyzer_path(""),
                env=my_env,
            )
            print_subprocess_results(p)

            print(f"tagging {offline_tag} as latest")
            p = subprocess.run(
                args=["docker", "tag", offline_tag, f"{OFFLINE_BASE_NAME}:latest"],
                cwd=analyzer_path(""),
            )
            print_subprocess_results(p)

    print("Done")
    print("you can now run docker-compose")


if __name__ == "__main__":
    setup_dockers()
