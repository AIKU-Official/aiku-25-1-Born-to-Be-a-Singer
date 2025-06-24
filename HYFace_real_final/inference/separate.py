import argparse
import os
from spleeter.separator import Separator

parser = argparse.ArgumentParser(description="오디오 파일에서 보컬을 분리합니다.")
parser.add_argument('--source', type=str, required=True, help='분리할 원본 오디오 파일 경로')
parser.add_argument('--output', type=str, default='inference/separated', help='분리된 파일들을 저장할 폴더')
args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output, exist_ok=True)

print(f"Spleeter로 '{args.source}' 파일 분리를 시작합니다...")

separator = Separator('spleeter:2stems')
separator.separate_to_file(args.source, args.output)

base_name = os.path.splitext(os.path.basename(args.source))[0]
vocals_path = os.path.join(args.output, base_name, 'vocals.wav')

if os.path.exists(vocals_path):
    print(f"분리 완료. 보컬 파일: {vocals_path}")
else:
    print(f"오류: 보컬 파일 분리에 실패했습니다.")