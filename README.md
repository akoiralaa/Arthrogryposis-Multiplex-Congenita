# Arthogyropsis Multiplex Congentia

Codebase for a fully automated, AI-powered tool that scores the SHAPE-UP assessment using Mediapipe Holistic, integrating pose, hand, and face landmarks to evaluate upper limb function in children with Arthrogryposis Multiplex Congenita (AMC). This system enhances scoring consistency, speed, and objectivity for clinical video assessments.

## Project Overview

The SHAPE-UP assessment evaluates how children with AMC perform daily upper-limb tasks. Our Holistic-based solution:
- Segments tasks from continuous video.
- Extracts upper body landmarks (shoulder, elbow, wrist, hand, face) using Mediapipe Holistic.
- Analyzes movement quality and compensatory strategies (e.g., trunk lean, bimanual use).
- Assigns SHAPE-UP scores automatically (0–2 scale).
- Outputs a summary CSV per task, ready for clinician review.

## Project Structure
```
shapeup-holistic/
├── run_shapeup_scoring.py        # Entry point: run full scoring pipeline
├── score_all_tasks.py            # Runs all 8 task scoring modules
├── tasks/                        # Per-task scoring logic (1 script per task)
│   ├── pick_up_cheerio.py
│   ├── bring_to_mouth.py
│   ├── ...
├── utils/
│   ├── common_landmarks.py       # Landmark indices & helper methods
│   ├── write_summary.py          # Output CSV formatting
├── models/                       # (Optional) Pretrained models
├── data/                         # Input videos and scoring outputs
├── requirements.txt              # Dependencies
└── README.md                     # Project overview

```

## Technologies Used

- Python
- OpenCV
- MediaPipe / OpenPose (for pose estimation)
- TensorFlow / PyTorch (for modeling)
- Pandas / NumPy
- Scikit-learn (optional for evaluation)
- Jupyter Notebooks (for debugging and visualization)


### Prerequisites

```bash
pip install -r requirements.txt
```
Make sure Mediapipe and OpenCV are properly installed. A GPU is recommended for real-time inference.

## Run The Pipeline
```
python run_shapeup_scoring.py --video_path path/to/input_video.mp4
```


This will:
- Load the video.
- Run holistic landmark extraction.
- Automatically segment and score all SHAPE-UP tasks.
- Outout a CSV file with task-wise scores.


## SHAPE-UP Scoring Criteria
```
+-------+-------------------------------------------------------------------------------------+
| Score | Description                                                                         |
+-------+-------------------------------------------------------------------------------------+
| 0     | Cannot complete the task without help; uses assistive tools or major compensations. |
+-------+-------------------------------------------------------------------------------------+
| 1     | Completes independently with significant compensation or adapted equipment.         |
+-------+-------------------------------------------------------------------------------------+
| 2     | Completes with no help and minimal compensation.                                    |
+-------+-------------------------------------------------------------------------------------+
```

## Evaluation
Model evaluation done using accuracy, F1-Score, and agreement with Clinicial ratings. 

## Contributors
- Abhie Koirala
- Anusha Marella
- Rahul Chaudhary
- Ishani Tamma

## License
Licensed under MIT license. 
