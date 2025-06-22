from utils.pose_utils import calculate_angle

def bicep_curl_tracking(landmarks, mode):
    if mode == "Right Arm":
        shoulder = [landmarks[11].x, landmarks[11].y]
        elbow = [landmarks[13].x, landmarks[13].y]
        wrist = [landmarks[15].x, landmarks[15].y]
    else:
        shoulder = [landmarks[12].x, landmarks[12].y]
        elbow = [landmarks[14].x, landmarks[14].y]
        wrist = [landmarks[16].x, landmarks[16].y]
    return calculate_angle(shoulder, elbow, wrist)

def both_arm_tracking(landmarks):
    l_angle = calculate_angle([landmarks[12].x, landmarks[12].y],
                              [landmarks[14].x, landmarks[14].y],
                              [landmarks[16].x, landmarks[16].y])
    r_angle = calculate_angle([landmarks[11].x, landmarks[11].y],
                              [landmarks[13].x, landmarks[13].y],
                              [landmarks[15].x, landmarks[15].y])
    return (l_angle + r_angle) / 2
