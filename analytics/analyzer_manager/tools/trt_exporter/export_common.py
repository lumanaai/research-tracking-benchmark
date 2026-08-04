import os
import platform
import re
import subprocess
from enum import Enum
from typing import Dict, Optional


class HostType(str, Enum):
    ORIN_JP5 = "orin-jp5"
    ORIN_JP6 = "orin-jp6"
    XAVIER = "xavier"
    CPU = "cpu"
    CUDA = "cuda"
    L4 = "l4"
    RTX3080 = "rtx3080"
    RTX4060 = "rtx4060"
    RTX4070 = "rtx4070bw"
    RTX4000ADA = "rtx4000ada"
    RTX4000PRO = "rtx4000pro"
    RTX2000PRO = "rtx2000pro"
    NONE = ""


def is_jetson_platform(host: HostType) -> bool:
    return host in [HostType.ORIN_JP5, HostType.ORIN_JP6, HostType.XAVIER]


def is_x86_cuda_platform(host: HostType) -> bool:
    return not is_jetson_platform(host) and host != HostType.CPU


def get_nvidia_smi() -> Optional[Dict[str, str]]:
    """
    Parse output of nvidia-smi into a python dictionary.
    This is very basic!
    """
    try:
        sp = subprocess.Popen(["nvidia-smi", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        return None

    out_bytes = sp.communicate()
    out_str = out_bytes[0].decode("UTF-8")
    out_list = out_str.split("\n")
    res_dict = {}
    for item in out_list:
        try:
            key_, val = item.split(":")
            key_, val = key_.strip(), val.strip()
            res_dict[key_] = val
        except ValueError:
            pass
    return res_dict


def get_platform_data() -> Dict[str, str]:
    return {
        "system": platform.system(),
        "arch": platform.architecture(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": get_processor_info(),
    }


def get_gpu_info() -> Optional[Dict[str, str]]:
    try:
        sp = subprocess.Popen(["nvidia-smi", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {"GPU": "True"}

    except FileNotFoundError:
        return None


def get_processor_info():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)
    return ""


def get_system_from_lshw():
    command = "lshw -short"
    all_info = subprocess.check_output(command, shell=True).decode().strip()
    system_desc = ""
    for line in all_info.split("\n"):
        if "system" in line:
            system_desc = line.replace("system", "").strip()
            break
    return system_desc


def get_jetpack_version():
    try:
        command = "cat /etc/nv_tegra_release"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        ver = re.search(r"#\sR(\w+)", all_info).group(1)
        rev = re.search(r"REVISION:\s*([\d\.]+)", all_info).group(1)
        return f"{ver}.{rev}"
    except subprocess.CalledProcessError:
        pass
    return ""


def determine_host(verbose: bool = False) -> HostType:
    smi_res = get_nvidia_smi()
    platform_data = get_platform_data()
    system_name = get_system_from_lshw()
    if verbose:
        import pprint

        print("##############   platform results: ##################")
        pprint.pprint(platform_data)
        print("##############   SMI results: ##################")
        pprint.pprint(smi_res)

    if "tegra" in platform_data["platform"]:
        if os.path.exists("/etc/avt_tegra_release"):
            with open("/etc/avt_tegra_release", "r") as file:
                # expected format: D115_ORIN_LUMANA-R3.2.0.6.1.0
                version_line = file.readline().strip()
                major_version = version_line.split(".")[3]
                if major_version == "5":
                    return HostType.ORIN_JP5
                elif major_version == "6":
                    return HostType.ORIN_JP6

        # in case the file doesn't exist or parsing issue
        os_version = platform_data["platform"].split("-")[1]
        if os_version.startswith("5.10"):
            return HostType.ORIN_JP5
        elif os_version.startswith("5.15"):
            return HostType.ORIN_JP6
        rev = [int(s) for s in platform_data["processor"].split() if s.isdigit()]
        if rev:
            if rev[0] == 1:
                return HostType.ORIN_JP5
            else:
                return HostType.XAVIER
        else:
            return HostType.ORIN_JP6
    elif smi_res is not None:
        ret_type = HostType.CUDA
        gpu_name = smi_res.get("Product Name", "")
        if "RTX 4000 Ada" in gpu_name:
            ret_type = HostType.RTX4000ADA
        elif "GeForce RTX 3080" in gpu_name:
            ret_type = HostType.RTX3080
        elif "GeForce RTX 4060" in gpu_name:
            ret_type = HostType.RTX4060
        elif "GeForce RTX 4070" in gpu_name:
            ret_type = HostType.RTX4070
        elif "L4" in gpu_name:
            ret_type = HostType.L4
        elif "NVIDIA RTX PRO 4000 Blackwell" in gpu_name:
            ret_type = HostType.RTX4000PRO
        elif "NVIDIA RTX PRO 2000 Blackwell" in gpu_name:
            ret_type = HostType.RTX2000PRO
        return ret_type
    else:
        return HostType.CPU


in_docker = True if os.getenv("IN_DOCKER") and os.getenv("IN_DOCKER") == "True" else False
export_dir = "/exports"
