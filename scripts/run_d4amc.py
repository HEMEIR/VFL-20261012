"""Single-run dataset_04_amc to verify the scenario override fixes it to LRA ~0.668.

    <python> scripts/run_d4amc.py
"""

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import loadConfig  # noqa: E402
from core.experiment import runExperiment  # noqa: E402


def main():
    dc = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "defense_purified_reassign_sym.yaml").read_text(encoding="utf-8")
    )
    from defense import buildPartyADefense
    d = buildPartyADefense(dc["defense"])
    d.applyDatasetOverrideByName("dataset_04")
    print("  dataset_04 注入后 rate=%s" % d.partyDropoutRate)
    d._applyScenarioOverrides("none", "amc")
    print("  none|amc 覆盖后 rate=%s (期望 0.2)" % d.partyDropoutRate)

    cfg = loadConfig(
        PROJECT_ROOT / "configs" / "scenarios" / "dataset_04_amc.yaml",
        {
            "training": {"epochs": 100, "device": "cuda", "progressEnabled": False},
            "output": {"directory": str(PROJECT_ROOT / "outputs" / "probe_d4amc")},
        },
        enforceFormalLock=False,
        defenseConfig=dc,
    )
    runExperiment(cfg, PROJECT_ROOT)
    v = yaml.safe_load(
        (PROJECT_ROOT / "outputs" / "probe_d4amc" / "metrics.json").read_text(encoding="utf-8")
    )["validationResults"]
    lra = (v.get("labelInference") or {}).get("labelRecoveryAccuracy")
    print("  d04/amc: acc=%.4f lra=%.4f (期望 ~0.668)" % (v["mainTask"]["accuracy"], lra))


if __name__ == "__main__":
    main()
