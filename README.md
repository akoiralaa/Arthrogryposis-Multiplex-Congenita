# Arthogyropsis Multiplex Congentia

Codebase for a fully automated, AI-powered tool that scores the SHAPE-UP assessment using Mediapipe Holistic, integrating pose, hand, and face landmarks to evaluate upper limb function in children with Arthrogryposis Multiplex Congenita (AMC). This system enhances scoring consistency, speed, and objectivity for clinical video assessments.

## Project Overview

The SHAPE-UP scored manually based on videos of subjects performing specific tasks. This project aims to automate that process by:
- Analyzing video footage of subjects performing tasks.
- Extracting key joint movements using pose estimation.
- Classifying performance according to a defined scoring rubric (0–2 scale).
- Outputting predicted scores aligned with clinical assessment guidelines.

## Project Structure
```
shapeup-ai/
  ├── data/                # Raw and processed data
  ├── notebooks/           # Jupyter notebooks for EDA and prototyping
  ├── models/              # Trained models, weights, checkpoints
  ├── scripts/             # Python scripts (training, inference, scoring)
  ├── utils/               # Utility functions (visualization, helpers)
  ├── app/                 # (Optional) Streamlit or Flask app
  ├── README.md             # Project overview
  └── requirements.txt      # Python dependencies
```

## Technologies Used

- Python
- OpenCV
- MediaPipe / OpenPose (for pose estimation)
- TensorFlow / PyTorch (for modeling)
- Pandas / NumPy
- Scikit-learn


### Prerequisites

```bash
pip install -r requirements.txt
```

## Run The Pipeline
```
python src/pipeline.py --video_path data/videos/sample1.mp4
```

## Model Training 
```To train the scoring model
python src/train_model.py --config configs/default.yaml
```

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
