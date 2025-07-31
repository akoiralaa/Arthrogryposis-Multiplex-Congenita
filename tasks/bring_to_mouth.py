import numpy as np
from utils.common_landmarks import POSE_LANDMARKS, calc_angle

def detect_used_arm(pose_landmarks):
    mouth = pose_landmarks[POSE_LANDMARKS['mouth']]
    lwrist = pose_landmarks[POSE_LANDMARKS['left_wrist']]
    rwrist = pose_landmarks[POSE_LANDMARKS['right_wrist']]

    dist_l = np.linalg.norm(np.array([lwrist.x, lwrist.y]) - np.array([mouth.x, mouth.y]))
    dist_r = np.linalg.norm(np.array([rwrist.x, rwrist.y]) - np.array([mouth.x, mouth.y]))

    return 'left' if dist_l < dist_r else 'right'

def detect_bimanual(pose_landmarks):
    lwrist = pose_landmarks[POSE_LANDMARKS['left_wrist']]
    rwrist = pose_landmarks[POSE_LANDMARKS['right_wrist']]
    distance = np.linalg.norm(np.array([lwrist.x, lwrist.y]) - np.array([rwrist.x, rwrist.y]))
    return distance < 0.1

def detect_and_score(results):
    if not results.pose_landmarks:
        return []

    pose = results.pose_landmarks.landmark
    used_arm = detect_used_arm(pose)

    wrist = pose[POSE_LANDMARKS[f'{used_arm}_wrist']]
    elbow = pose[POSE_LANDMARKS[f'{used_arm}_elbow']]
    shoulder = pose[POSE_LANDMARKS[f'{used_arm}_shoulder']]

    elbow_angle = calc_angle(
        [shoulder.x, shoulder.y],
        [elbow.x, elbow.y],
        [wrist.x, wrist.y]
    )

    shoulder_angle = calc_angle(
        [elbow.x, elbow.y],
        [shoulder.x, shoulder.y],
        [pose[POSE_LANDMARKS['mouth']].x, pose[POSE_LANDMARKS['mouth']].y]
    )

    elbow_motion = 'limited' if elbow_angle < 90 else 'some'
    shoulder_motion = 'limited' if shoulder_angle < 45 else 'some'

    score = 2 if elbow_angle > 90 and shoulder_angle > 45 else 1
    compensation = 'bimanual' if detect_bimanual(pose) else 'none'

    return [{
        'task_name': 'BringToMouth',
        'arm': used_arm,
        'score': score,
        'prehension': 'precision',
        'wrist_motion': 'some',
        'forearm_motion': 'some',
        'elbow_motion': elbow_motion,
        'shoulder_motion': shoulder_motion,
        'compensation': compensation,
        'adaptive_equipment': 'none',
        'amc_severity': 'mild' if score == 2 else 'moderate'
    }]
