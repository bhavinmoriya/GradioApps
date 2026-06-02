from pydub import AudioSegment, silence
import sys
import os

def trim_silence_mp3(input_path, output_path, silence_len=500, silence_thresh=-40, keep_silence=50):
    audio = AudioSegment.from_mp3(input_path)

    chunks = silence.split_on_silence(
        audio,
        min_silence_len=silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )

    if not chunks:
        print("No non-silent audio found.")
        return

    trimmed = AudioSegment.empty()
    for chunk in chunks:
        trimmed += chunk

    trimmed.export(output_path, format="mp3")
    print(f"Saved trimmed file to: {output_path}")

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python trim_mp3.py input.mp3 output.mp3")
#         sys.exit(1)

#     input_file = sys.argv[1]
#     output_file = sys.argv[2]

#     if not os.path.exists(input_file):
#         print(f"Input file not found: {input_file}")
#         sys.exit(1)

#     trim_silence_mp3(input_file, output_file)

# Test with explicit paths
input_mp3_path = "/content/abc.mp3" # Assuming this file exists from previous steps
output_mp3_path = "/content/abc_trimmed.mp3"

if os.path.exists(input_mp3_path):
    trim_silence_mp3(input_mp3_path, output_mp3_path)
else:
    print(f"Input file not found: {input_mp3_path}. Please ensure it is uploaded or created.")
