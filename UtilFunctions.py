import cv2 as cv
import numpy as np
import mediapipe as mp
import torch


def extract_landmarks(frame, mp_holistic, drawHands=False, drawPose=False, drawFace=False):
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the image
    results = mp_holistic.process(image)

    # Get landmarks from results if available, otherwise set to zeros
    face_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in
                               results.face_landmarks.landmark]).flatten() if drawFace and results.face_landmarks else np.zeros(
        (468 * 3,))
    pose_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in
                               results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(
        (33 * 3,))
    left_hand_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in
                                    results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
        (21 * 3,))
    right_hand_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in
                                     results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        (21 * 3,))

    # Drawing landmarks on image
    if drawFace and results.face_landmarks:
        for landmark in results.face_landmarks.landmark:
            cv.circle(frame, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 2, (255, 0, 0), 2)
    if drawHands and results.left_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.left_hand_landmarks,
                                                  mp.solutions.holistic.HAND_CONNECTIONS)
    if drawHands and results.right_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.right_hand_landmarks,
                                                  mp.solutions.holistic.HAND_CONNECTIONS)
    if drawPose and results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)

    # Concatenate all landmarks into a single array
    all_landmarks = np.concatenate((face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks))
    return all_landmarks, frame


def extract_landmarks_holistic(frame, mp_holistic, validNeeded=0):
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the image
    results = mp_holistic.process(image)

    # Get landmarks from results if available, otherwise set to zeros
    face_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in
                               results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(
        (468 * 3,))
    pose_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in
                               results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(
        (33 * 3,))
    left_hand_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in
                                    results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
        (21 * 3,))
    right_hand_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in
                                     results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        (21 * 3,))

    # Drawing landmarks on image
    if results.face_landmarks:
        for landmark in results.face_landmarks.landmark:
            cv.circle(frame, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 2, (255, 0, 0), 2)
    if results.left_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.left_hand_landmarks,
                                                  mp.solutions.holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.right_hand_landmarks,
                                                  mp.solutions.holistic.HAND_CONNECTIONS)
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)

    # Concatenate all landmarks into a single array
    all_landmarks = np.concatenate((face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks))

    # If validNeeded is >0 and not enough landmarks are valid, set all_landmarks as None
    if validNeeded > 0:
        if results.pose_landmarks is None:
             all_landmarks = None
        valid_hands = 0
        if results.left_hand_landmarks is not None:
            valid_hands += 1
        if results.right_hand_landmarks is not None:
            valid_hands += 1
        if valid_hands < validNeeded:
            all_landmarks = None
    return all_landmarks, frame

def extract_landmarks_hand_pose(frame, mp_hands, mp_pose, validNeeded=0, shouldDrawOnImage=False):
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    if(mp_hands):
        # Process the image
        hands_results = mp_hands.process(image)
        # Prepare an array for hands landmarks
        left_hand_landmarks = None
        right_hand_landmarks = None
        # Check if any hand is detected
        if hands_results.multi_hand_landmarks:
            # Assuming first hand is right and second is left
            right_hand_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in
                                             hands_results.multi_hand_landmarks[0].landmark]).flatten()
            if len(hands_results.multi_hand_landmarks) > 1:
                left_hand_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in
                                                hands_results.multi_hand_landmarks[1].landmark]).flatten()

        # If any hand is not detected, append zeros
        right_hand_landmarks = right_hand_landmarks if right_hand_landmarks is not None else np.zeros((21 * 3,))
        left_hand_landmarks = left_hand_landmarks if left_hand_landmarks is not None else np.zeros((21 * 3,))

    if mp_pose:
        # Process the image
        pose_results = mp_pose.process(image)
        # Get landmarks from results if available, otherwise set to zeros
        pose_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in
                                   pose_results.pose_landmarks.landmark]).flatten() if pose_results.pose_landmarks else np.zeros(
            (33 * 3,))

    if mp_hands and mp_pose:
        # Concatenate all landmarks into a single array
        all_landmarks = np.concatenate((pose_landmarks, right_hand_landmarks, left_hand_landmarks))
    elif mp_hands:
        # Concatenate all landmarks into a single array
        all_landmarks = np.concatenate((right_hand_landmarks, left_hand_landmarks))
    elif mp_pose:
        # Concatenate all landmarks into a single array
        all_landmarks = pose_landmarks

    if shouldDrawOnImage:
        # Drawing landmarks on image
        if mp_hands and hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks,
                                                          mp.solutions.hands.HAND_CONNECTIONS)
        if mp_pose and pose_results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks,
                                                      mp.solutions.pose.POSE_CONNECTIONS)

    # If validNeeded is > 0 and not all landmarks are valid, set all_landmarks as None
    if validNeeded:
        if all_landmarks is None:
            print("is schon None")
        if mp_hands and (hands_results.multi_hand_landmarks is None or len(hands_results.multi_hand_landmarks) < validNeeded):
            all_landmarks = None
        if mp_pose and pose_results.pose_landmarks is None:
            all_landmarks = None
    return all_landmarks, frame


def process_landmarks(landmarks):
    # Converts landmarks into a tensor and adds another dimension to fit 
    # the (batch_size, num_features). 
    landmarks = torch.tensor(landmarks)
    landmarks = landmarks.unsqueeze_(0).float()  # convert to (1, num_features)
    return landmarks


def predict_landmark_class(landmarks, model):
    # Use your classifier model to predict the landmark class.
    output = model(landmarks)
    _, pred = torch.max(output, 1)  # get the index of the max log-probability
    return pred.item()
