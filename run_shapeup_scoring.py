import sys
from score_all_tasks import process_video

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_shapeup_scoring.py path_to_video.mp4")
        return

    video_path = sys.argv[1]
    output_csv = "scoring_output.csv"

    print(f"Input video: {video_path}")
    process_video(video_path, output_csv=output_csv)

if __name__ == "__main__":
    main()
