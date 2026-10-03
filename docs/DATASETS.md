# Dataset shortlist for TRACE

| Dataset | Best use in this project | Limitation / terms |
| --- | --- | --- |
| Your own camera footage | Final detector and event evaluation | Obtain the right to process the footage; redact or protect images as required |
| COCO person class | Generic pretrained detector and pipeline smoke test | Natural images differ from fixed CCTV views; review the dataset terms |
| CrowdHuman | Occlusion and crowd robustness | Non-commercial research and educational use only according to its download terms: https://www.crowdhuman.org/download.html |
| MOT17 | Tracking and ID stability checks | Benchmark sequences are short and pedestrian-focused; archived download page: https://motchallenge.net/data/MOT17/ |
| UCF-Crime | Later video-level anomaly research such as fighting or robbery | It has 13 anomaly categories and video-level training labels; it is not a direct restricted-zone-entry dataset: https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/ |

The most important dataset is a held-out set from the intended cameras. Public data can improve generalization, but it cannot specify your zone boundaries, dwell policy, camera angle, lighting, or acceptable false-alert rate. Capture negative examples deliberately: people who approach the boundary, people who pass through a permitted area, empty scenes, shadows, reflections, and crowded crossings.

