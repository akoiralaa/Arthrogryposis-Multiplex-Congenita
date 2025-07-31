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

def detect_throw_posture(pose_landmarks, used_arm):
    wrist = pose_landmarks[POSE_LANDMARKS[f'{used_arm}_wrist']]
    shoulder = pose_landmarks[POSE_LANDMARKS[f'{used_arm}_shoulder']]
    hip = pose_landmarks[POSE_LANDMARKS[f'{used_arm}_hip']]
    
    vertical_range = abs(wrist.y - shoulder.y)
    torso_bend = abs(shoulder.x - hip.x)
    
    return vertical_range > 0.1 and torso_bend > 0.05

def detect_and_score(results):
    if not results.pose_landmarks:
        return []

    pose = results.pose_landmarks.landmark
    used_arm = detect_used_arm(pose)

    wrist = pose[POSE_LANDMARKS[f'{used_arm}_wrist']]
    elbow = pose[POSE_LANDMARKS[f'{used_arm}_elbow']]
    shoulder = pose[POSE_LANDMARKS[f'{used_arm}_shoulder']]

    elbow_angle = calc_angle([shoulder.x, shoulder.y], [elbow.x, elbow.y], [wrist.x, wrist.y])
    shoulder_angle = calc_angle([elbow.x, elbow.y], [shoulder.x, shoulder.y], [pose[POSE_LANDMARKS['hip']].x, pose[POSE_LANDMARKS['hip']].y])

    elbow_motion = 'limited' if elbow_angle < 100 else 'good'
    shoulder_motion = 'limited' if shoulder_angle < 70 else 'good'

    throw_like_posture = detect_throw_posture(pose, used_arm)
    score = 2 if elbow_motion == 'good' and shoulder_motion == 'good' and throw_like_posture else 1

    return [{
        'task_name': 'ThrowBall',
        'arm': used_arm,
        'score': score,
        'prehension': 'power',
        'wrist_motion': 'some',
        'forearm_motion': 'some',
        'elbow_motion': elbow_motion,
        'shoulder_motion': shoulder_motion,
        'compensation': 'torso lean' if throw_like_posture else 'none',
        'adaptive_equipment': 'none',
        'amc_severity': 'mild' if score == 2 else 'moderate'
    }]
