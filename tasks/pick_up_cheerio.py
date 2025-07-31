import numpy as np
from utils.common_landmarks import POSE_LANDMARKS, calc_angle

def detect_used_arm(pose_landmarks):
    mouth = pose_landmarks[POSE_LANDMARKS['mouth']]
    lwrist = pose_landmarks[POSE_LANDMARKS['left_wrist']]
    rwrist = pose_landmarks[POSE_LANDMARKS['right_wrist']]

    dist_l = np.linalg.norm(np.array([lwrist.x, lwrist.y]) - np.array([mouth.x, mouth.y]))
    dist_r = np.linalg.norm(np.array([rwrist.x, rwrist.y]) - np.array([mouth.x, mouth.y]))
    
    return 'left' if dist_l < dist_r else 'right'

def detect_precision_grip(hand_landmarks):
    if not hand_landmarks:
        return False
    # Tip IDs: Thumb (4), Index (8)
    thumb_tip = np.array([hand_landmarks[4].x, hand_landmarks[4].y])
    index_tip = np.array([hand_landmarks[8].x, hand_landmarks[8].y])
    distance = np.linalg.norm(thumb_tip - index_tip)
    return distance < 0.05  # Precision grip if thumb/index close

def detect_and_score(results):
    if not results.pose_landmarks:
        return []

    pose = results.pose_landmarks.landmark
    left_hand = results.left_hand_landmarks.landmark if results.left_hand_landmarks else None
    right_hand = results.right_hand_landmarks.landmark if results.right_hand_landmarks else None

    used_arm = detect_used_arm(pose)
    hand_lm = left_hand if used_arm == 'left' else right_hand

    wrist = pose[POSE_LANDMARKS[f'{used_arm}_wrist']]
    elbow = pose[POSE_LANDMARKS[f'{used_arm}_elbow']]
    shoulder = pose[POSE_LANDMARKS[f'{used_arm}_shoulder']]

    elbow_angle = calc_angle(
        [shoulder.x, shoulder.y],
        [elbow.x, elbow.y],
        [wrist.x, wrist.y]
    )

    elbow_motion = 'limited' if elbow_angle < 90 else 'some'
    score = 2 if elbow_angle > 90 else 1
    grip_type = 'precision' if detect_precision_grip(hand_lm) else 'gross'

    return [{
        'task_name': 'PickUpCheerio',
        'arm': used_arm,
        'score': score,
        'prehension': grip_type,
        'wrist_motion': 'some',
        'forearm_motion': 'some',
        'elbow_motion': elbow_motion,
        'shoulder_motion': 'some',
        'compensation': 'none',
        'adaptive_equipment': 'none',
        'amc_severity': 'mild' if score == 2 else 'moderate'
    }]
