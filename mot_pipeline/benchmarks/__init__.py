from mot_pipeline.benchmarks.cityflow import CityFlowBenchmark
from mot_pipeline.benchmarks.fasttracker_bench import FastTrackerBenchBenchmark
from mot_pipeline.benchmarks.lumana_benchmark import LumanaBenchmark
from mot_pipeline.benchmarks.trafficmot import TrafficMOTBenchmark
from mot_pipeline.benchmarks.ua_detrac import UADetracBenchmark

BENCHMARKS = {
    "fasttracker_bench": FastTrackerBenchBenchmark,
    "ua_detrac": UADetracBenchmark,
    "trafficmot": TrafficMOTBenchmark,
    "cityflow": CityFlowBenchmark,
    "lumana_benchmark": LumanaBenchmark,
}


def get_benchmark(name: str, **kwargs):
    if name not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark '{name}'. Choose from: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name](**kwargs)
