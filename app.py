import gradio as gr
import os
import shutil
import subprocess
import sys
import cv2
from datetime import datetime

# 얼굴 추출 함수 추가
def extract_face_or_image(input_path, output_image_path):
    ext = os.path.splitext(input_path)[-1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        shutil.copyfile(input_path, output_image_path)
        print(f"[INFO] 이미지 파일을 그대로 사용: {output_image_path}")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    vidcap = cv2.VideoCapture(input_path)
    found = False

    while True:
        success, frame = vidcap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            cv2.imwrite(output_image_path, frame)
            print(f"[INFO] 얼굴이 검출된 프레임을 {output_image_path} 에 저장했습니다.")
            found = True
            break

    vidcap.release()

    if not found:
        raise RuntimeError("영상에서 얼굴을 찾을 수 없습니다.")

# Wav2Lip 실행 함수
def run_wav2lip(face_path, audio_path, output_path):
    python_path = sys.executable
    print("[INFO] Running Wav2Lip...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2"
    result = subprocess.run(
        [python_path, "run_wav2lip.py",
         "--face", face_path,
         "--audio", audio_path,
         "--outfile", output_path],
        cwd="Wav2Lip",
        env=env
    )
    if result.returncode != 0:
        raise RuntimeError("Wav2Lip failed to run properly")

# HYFace 실행 함수 (이제 항상 이미지가 들어감)
def run_hyface(target_image_path, source_audio_path, output_audio_path, output_vocal_path):
    print("[INFO] Running HYFace...")
    python_path = sys.executable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    result = subprocess.run(
        [python_path, "run_hyface.py",
         "--target", target_image_path,
         "--source", source_audio_path,
         "--output_path", output_audio_path,
         "--output_vocal_path", output_vocal_path],
        cwd="HYFace_real_final",
        env=env
    )
    if result.returncode != 0:
        raise RuntimeError("HYFace failed to run properly")

def replace_audio_in_video(video_path, new_audio_path, output_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")
    if not os.path.exists(new_audio_path):
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {new_audio_path}")

    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", new_audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    print(f"[INFO] FFmpeg 실행: {' '.join(command)}")
    subprocess.run(command, check=True)
    return output_path

# 전체 파이프라인
def inference(video_or_image_path, audio_path, final_output_path):
    hyface_output_path = final_output_path.replace(".mp4", "_hyface.wav")
    hyface_vocal_path = final_output_path.replace(".mp4", "_hyface_vocal.wav")
    wav2lip_path = final_output_path.replace(".mp4", "_wav2lip.mp4")

    # 얼굴 이미지 추출 (이미지/영상 구분 자동)
    face_image_path = final_output_path.replace(".mp4", "_face.jpg")
    extract_face_or_image(video_or_image_path, face_image_path)

    run_hyface(target_image_path=face_image_path,
               source_audio_path=audio_path,
               output_audio_path=hyface_output_path,
               output_vocal_path=hyface_vocal_path)

    run_wav2lip(face_path=video_or_image_path,  # 영상 그대로 사용
                audio_path=hyface_vocal_path,
                output_path=wav2lip_path)

    replace_audio_in_video(wav2lip_path, hyface_output_path, final_output_path)

    for f in [hyface_output_path, hyface_vocal_path, wav2lip_path, face_image_path]:
        if os.path.exists(f):
            os.remove(f)

# Gradio에서 호출할 process 함수
def process(video_file, audio_file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join("/home/aikusrv03/lipsync/jobs", timestamp)
    os.makedirs(work_dir, exist_ok=True)

    video_or_image_path = os.path.join(work_dir, os.path.basename(video_file.name))
    audio_path = os.path.join(work_dir, os.path.basename(audio_file.name))

    shutil.copyfile(video_file.name, video_or_image_path)
    shutil.copyfile(audio_file.name, audio_path)

    output_path = os.path.join(work_dir, "final_output.mp4")
    inference(video_or_image_path, audio_path, output_path)
    return output_path

# Gradio UI
with gr.Blocks(
    css="""
    .gradio-container {
        text-align: center;
    }
    #main_column {
        align-items: center;
        justify-content: center;
    }
    """
) as demo:
    gr.Markdown("# 내가 가수가 될 상인가 🎤")
    gr.Markdown("본격 관상으로만 립싱크 영상 만들기")

    with gr.Column(elem_id="main_column"):
        video_input = gr.File(label="영상 or 이미지 업로드", file_types=[".mp4", ".mov", ".avi", ".mkv", ".jpg", ".jpeg", ".png"])
        video_preview = gr.Video(label="업로드 미리보기")

        audio_input = gr.File(label="음악 파일 업로드", file_types=[".mp3", ".wav", ".aac"])
        audio_preview = gr.Audio(label="업로드한 오디오 미리듣기")

        generate_btn = gr.Button("🎬 영상 생성하기")
        result_video = gr.Video(label="🎧 결과 영상")

    video_input.change(fn=lambda f: f.name if f else None, inputs=video_input, outputs=video_preview)
    audio_input.change(fn=lambda f: f.name if f else None, inputs=audio_input, outputs=audio_preview)

    generate_btn.click(fn=process, inputs=[video_input, audio_input], outputs=result_video)

demo.queue()

if __name__ == "__main__":
    demo.launch(share=True)
