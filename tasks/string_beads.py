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

def detect_fine_pinch(hand_landmarks):
    if not hand_landmarks:
        return False
    thumb_tip = np.array([hand_landmarks[4].x, hand_landmarks[4].y])
    index_tip = np.array([hand_landmarks[8].x, hand_landmarks[8].y])
    dist = np.linalg.norm(thumb_tip - index_tip)
    return dist < 0.04

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

    pinch_detected = detect_fine_pinch(hand_lm)
    score = 2 if elbow_angle > 90 and shoulder_angle > 45 and pinch_detected else 1

    return [{
        'task_name': 'StringBeads',
        'arm': used_arm,
        'score': score,
        'prehension': 'precision' if pinch_detected else 'none',
        'wrist_motion': 'fine',
        'forearm_motion': 'some',
        'elbow_motion': elbow_motion,
        'shoulder_motion': shoulder_motion,
        'compensation': 'none',
        'adaptive_equipment': 'none',
        'amc_severity': 'mild' if score == 2 else 'moderate'
    }]
