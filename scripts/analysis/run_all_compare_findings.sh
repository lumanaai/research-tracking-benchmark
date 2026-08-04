# GT oracle dets
.venv/bin/python scripts/analysis/compare_findings.py \
  --detector-id gt_vehicles --format both \
  --out results/comparisons/gt_vehicles.md

# GT oracle dets FPS=5
.venv/bin/python scripts/analysis/compare_findings.py \
  --detector-id gt_vehicles --target-fps 5 --format both \
  --out results/comparisons/gt_vehicles_fps5.md

# GT oracle dets FPS=10
.venv/bin/python scripts/analysis/compare_findings.py \
  --detector-id gt_vehicles --target-fps 10 --format both \
  --out results/comparisons/gt_vehicles_fps10.md

# native FPS
.venv/bin/python scripts/analysis/compare_findings.py \
  --detector-id yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles --format both \
  --out results/comparisons/yolov8m_expert_eff.md

# FPS=10
.venv/bin/python scripts/analysis/compare_findings.py \
  --detector-id yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles --target-fps 10 --format both \
  --out results/comparisons/yolov8m_expert_eff_fps10.md

# FPS=5
.venv/bin/python scripts/analysis/compare_findings.py \
  --detector-id yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles --target-fps 5 --format both \
  --out results/comparisons/yolov8m_expert_eff_fps5.md