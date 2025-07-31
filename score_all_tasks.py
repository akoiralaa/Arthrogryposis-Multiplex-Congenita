import cv2
import pandas as pd
import mediapipe as mp
from utils.write_summary import write_summary_csv
from utils.common_landmarks import extract_frames_with_results

# === Import all 8 real scoring scripts ===
from tasks.pick_up_cheerio import detect_and_score as score_cheerio
from tasks.bring_to_mouth import detect_and_score as score_mouth
from tasks.grasp_water_bottle import detect_and_score as score_water
from tasks.string_beads import detect_and_score as score_beads
from tasks.draw_scribble import detect_and_score as score_scribble
from tasks.throw_ball import detect_and_score as score_throw
from tasks.put_on_vest import detect_and_score as score_vest
from tasks.pull_down_shorts import detect_and_score as score_shorts

# === Mediapipe Holistic ===
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# === Main Processing ===
def process_video(video_path, output_csv='scoring_output.csv'):
    cap = cv2.VideoCapture(video_path)
    holistic = mp_holistic.Holistic(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("🔍 Extracting results using Mediapipe Holistic...")
    results_per_frame = extract_frames_with_results(cap, holistic)
    holistic.close()
    cap.release()

    print("Running all scoring modules...")

    all_rows = []
    all_rows += score_cheerio(results_per_frame)
    all_rows += score_mouth(results_per_frame)
    all_rows += score_water(results_per_frame)
    all_rows += score_beads(results_per_frame)
    all_rows += score_scribble(results_per_frame)
    all_rows += score_throw(results_per_frame)
    all_rows += score_vest(results_per_frame)
    all_rows += score_shorts(results_per_frame)

    write_summary_csv(all_rows, output_csv)
    print(f"Results written to {output_csv}")
