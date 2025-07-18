import argparse
parser = argparse.ArgumentParser()
# Model Configuration:
parser.add_argument("--audio_dir_path", type=str, help="Directory for the tracks that will be included")
args = parser.parse_args()
print(args.audio_dir_path)