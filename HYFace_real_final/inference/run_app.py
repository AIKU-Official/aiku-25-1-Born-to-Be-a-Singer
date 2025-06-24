import os
import subprocess
import sys

# ===================================================================
#                          사용자 설정
# ===================================================================
# 변환할 원본 오디오 파일
SOURCE_AUDIO = "inference/taeyeon_reference.mp3"

# 목소리를 입힐 타겟 얼굴 이미지
TARGET_IMAGE = "inference/winter_black.jpg"

# 사용할 GPU 번호
GPU_ID = "0"
# ===================================================================


def run_command(command, env=None):
    """주어진 명령어를 실행하고 실시간으로 출력을 보여주는 함수"""
    print(f"\n[실행 명령어]: {' '.join(command)}\n")
    # Popen을 사용하여 서브프로세스를 만들고, 표준 출력을 파이프로 연결
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True, env=env)
    
    # 서브프로세스가 종료될 때까지 실시간으로 출력 읽기
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    process.wait() # 프로세스가 끝날 때까지 대기
    print("\n")
    
    if process.returncode != 0:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"오류: 명령어가 실패했습니다 (종료 코드: {process.returncode})")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit(1) # 오류 발생 시 전체 스크립트 종료

def main(args):
    # 1. 보컬 분리 스크립트 실행
    print("="*25)
    print("  1. 보컬 분리 시작")
    SOURCE_AUDIO = args.source
    

# 목소리를 입힐 타겟 얼굴 이미지

    TARGET_IMAGE = args.target
    print("="*25)
    separate_command = ["python3", "inference/separate.py", "--source", SOURCE_AUDIO]
    run_command(separate_command)

    # 2. 분리된 파일 경로 계산
    basename = os.path.basename(SOURCE_AUDIO)
    filename, _ = os.path.splitext(basename)
    vocals_path = os.path.join("inference", "separated", filename, "vocals.wav")
    inst_path = os.path.join("inference", "separated", filename, "accompaniment.wav")

    # 분리된 파일이 존재하는지 확인
    if not os.path.exists(vocals_path) or not os.path.exists(inst_path):
        print(f"오류: 분리된 파일({vocals_path} 또는 {inst_path})을 찾을 수 없습니다.")
        sys.exit(1)

    # 3. 음성 변환 스크립트 실행
    print("="*25)
    print("  2. 수정 HYFACE 모델 음성 변환 시작")
    print("="*25)
    
    # CUDA_VISIBLE_DEVICES 환경 변수 설정을 위해 현재 환경 복사
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    
    inference_command = [
        "python3", "inference/inference_app.py",
        "--source", vocals_path,
        "--target", TARGET_IMAGE,
        "--inst", inst_path,
        "--song_name", os.path.basename(SOURCE_AUDIO),
        "--singer", os.path.basename(TARGET_IMAGE),
        "--cuda", GPU_ID,
        "--output_path",args.output_path,
        "--output_vocal_path",args.output_vocal_path
    ]
    run_command(inference_command, env=my_env)


if __name__ == "__main__":
    # 필요한 파일들이 존재하는지 확인
    required_files = [SOURCE_AUDIO, TARGET_IMAGE, "inference/separate.py", "inference/inference.py"]
    for f in required_files:
        if not os.path.exists(f):
            print(f"오류: 필수 파일 '{f}'를 찾을 수 없습니다. 경로를 확인해주세요.")
            sys.exit(1)
            
    main()