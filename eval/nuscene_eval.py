from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes import NuScenes
import json

# Load dataset
nusc = NuScenes(
    version='v1.0-trainval',
    dataroot='\\thesis\\data\\nuscenes',
    verbose=True
)
result_path = 'results.json'
output_dir = './eval_output'

nusc_eval = NuScenesEval(
    nusc,
    config_name='detection_cvpr_2019',
    result_path=result_path,
    eval_set='val',
    output_dir=output_dir,
    verbose=True
)

nusc_eval.main()

with open("metrics_details.json") as f:
    data = json.load(f)

label_aps = data["label_aps"]

for cls, values in label_aps.items():
    # Average over distance thresholds
    ap = sum(values.values()) / len(values)
    print(f"{cls}: {round(ap*100, 2)}")
