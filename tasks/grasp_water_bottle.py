import numpy as np
from utils.common_landmarks import POSE_LANDMARKS, calc_angle

def detect_used_arm(pose_landmarks):
    lwrist = pose_landmarks[POSE_LANDMARKS['left_wrist']]
    rwrist = pose_landmarks[POSE_LANDMARKS['right_wrist']]
    lshoulder = pose_landmarks[POSE_LANDMARKS['left_shoulder']]
    rshoulder = pose_landmarks[POSE_LANDMARKS['right_shoulder']]

    dist_l = np.linalg.norm(np.array([lwrist.x, lwrist.y]) - np.array([lshoulder.x, lshoulder.y]))
    dist_r = np.linalg.norm(np.array([rwrist.x, rwrist.y]) - np.array([rshoulder.x, rshoulder.y]))

    return 'left' if dist_l < dist_r else 'right'

def detect_power_grip(hand_landmarks):
    if not hand_landmarks:
        return False
    # Check palm openness: distance between thumb base (2) and pinky base (17)
    thumb_base = np.array([hand_landmarks[2].x, hand_landmarks[2].y])
    pinky_base = np.array([hand_landmarks[17].x, hand_landmarks[17].y])
    spread = np.linalg.norm(thumb_base - pinky_base)
    return spread > 0.08

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

    elbow_angle = calc_angle([shoulder.x, shoulder.y], [elbow.x, elbow.y], [wrist.x, wrist.y])
    shoulder_angle = calc_angle([elbow.x, elbow.y], [shoulder.x, shoulder.y], [pose[POSE_LANDMARKS['mouth']].x, pose[POSE_LANDMARKS['mouth']].y])

    elbow_motion = 'limited' if elbow_angle < 90 else 'some'
    shoulder_motion = 'limited' if shoulder_angle < 45 else 'some'
    grip_type = 'power' if detect_power_grip(hand_lm) else 'none'

    score = 2 if elbow_angle > 90 and shoulder_angle > 45 and grip_type == 'power' else 1

    return [{
        'task_name': 'GraspWaterBottle',
        'arm': used_arm,
        'score': score,
        'prehension': grip_type,
        'wrist_motion': 'some',
        'forearm_motion': 'some',
        'elbow_motion': elbow_motion,
        'shoulder_motion': shoulder_motion,
        'compensation': 'none',
        'adaptive_equipment': 'none',
        'amc_severity': 'mild' if score == 2 else 'moderate'
    }]
