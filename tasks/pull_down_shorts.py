import numpy as np
from utils.common_landmarks import POSE_LANDMARKS, calc_angle

def detect_used_arm(pose_landmarks):
    lwrist = pose_landmarks[POSE_LANDMARKS['left_wrist']]
    rwrist = pose_landmarks[POSE_LANDMARKS['right_wrist']]
    lhip = pose_landmarks[POSE_LANDMARKS['left_hip']]
    rhip = pose_landmarks[POSE_LANDMARKS['right_hip']]

    dist_l = np.linalg.norm(np.array([lwrist.x, lwrist.y]) - np.array([lhip.x, lhip.y]))
    dist_r = np.linalg.norm(np.array([rwrist.x, rwrist.y]) - np.array([rhip.x, rhip.y]))

    return 'bilateral' if abs(dist_l - dist_r) < 0.05 else ('left' if dist_l < dist_r else 'right')

def detect_and_score(results):
    if not results.pose_landmarks:
        return []

    pose = results.pose_landmarks.landmark
    used_arm = detect_used_arm(pose)
    arms = ['left', 'right'] if used_arm == 'bilateral' else [used_arm]

    elbow_motion_list = []
    shoulder_motion_list = []

    for arm in arms:
        wrist = pose[POSE_LANDMARKS[f'{arm}_wrist']]
        elbow = pose[POSE_LANDMARKS[f'{arm}_elbow']]
        shoulder = pose[POSE_LANDMARKS[f'{arm}_shoulder']]
        hip = pose[POSE_LANDMARKS[f'{arm}_hip']]

        elbow_angle = calc_angle([shoulder.x, shoulder.y], [elbow.x, elbow.y], [wrist.x, wrist.y])
        shoulder_angle = calc_angle([elbow.x, elbow.y], [shoulder.x, shoulder.y], [hip.x, hip.y])

        elbow_motion_list.append('limited' if elbow_angle < 90 else 'some')
        shoulder_motion_list.append('limited' if shoulder_angle < 45 else 'some')

    elbow_motion = 'limited' if 'limited' in elbow_motion_list else 'some'
    shoulder_motion = 'limited' if 'limited' in shoulder_motion_list else 'some'

    score = 2 if elbow_motion == 'some' and shoulder_motion == 'some' else 1

    return [{
        'task_name': 'PullDownShorts',
        'arm': used_arm,
        'score': score,
        'prehension': 'none',
        'wrist_motion': 'some',
        'forearm_motion': 'some',
        'elbow_motion': elbow_motion,
        'shoulder_motion': shoulder_motion,
        'compensation': 'bimanual' if used_arm == 'bilateral' else 'none',
        'adaptive_equipment': 'none',
        'amc_severity': 'mild' if score == 2 else 'moderate'
    }]
