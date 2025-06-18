import os
import soundfile as sf
import matplotlib.pyplot as plt

def get_wav_durations(folder_path):
    durations = {}
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(".wav"):
            filepath = os.path.join(folder_path, filename)
            try:
                with sf.SoundFile(filepath) as f:
                    duration = len(f) / f.samplerate
                    durations[filename] = duration
            except RuntimeError as e:
                print(f"Skipping unreadable file {filename}: {e}")
    return durations

def plot_durations(durations, save_path=None):
    plt.figure(figsize=(12, 6))
    files = list(durations.keys())
    lengths = list(durations.values())

    plt.bar(files, lengths, color='#4B9CD3')
    plt.xlabel("Audio File", fontsize=12)
    plt.ylabel("Duration (seconds)", fontsize=12)
    plt.title("WAV File Duration Statistics", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=6)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    folder_path = "/mmu-audio-ssd/zhenghaorui/others/SD/Large-scale-diarization-dataset/exp/exp1/wavs"  # Replace with your folder path
    durations = get_wav_durations(folder_path)
    if durations:
        plot_durations(durations, save_path="wav_durations.png")
    else:
        print("No .wav files found.")
