import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

def load_speaker_info(json_paths):
    speaker_info = {}
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data = json.load(f)
            for entry in data:
                sid = entry['speaker_id']
                gender = entry['gender'].lower()
                language = entry['language'].lower()
                speaker_info[sid] = {'gender': gender, 'language': language}
    return speaker_info

def analyze_rttm_files(rttm_folder, speaker_info):
    gender_counts = {"male": 0, "female": 0, "mixed": 0}
    lang_counts = {"chinese": 0, "english": 0, "mixed": 0}

    for fname in os.listdir(rttm_folder):
        if not fname.endswith(".list"):
            continue
        speaker_ids = set()
        with open(os.path.join(rttm_folder, fname), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    speaker_ids.add(parts[2])

        genders = set()
        langs = set()
        for sid in speaker_ids:
            info = speaker_info.get(sid)
            if info:
                genders.add(info['gender'])
                langs.add(info['language'])

        # Gender classification
        if len(genders) == 1:
            gender_counts[list(genders)[0]] += 1
        elif len(genders) > 1:
            gender_counts["mixed"] += 1

        # Language classification
        if len(langs) == 1:
            lang = list(langs)[0]
            lang_counts[lang] += 1
        elif len(langs) > 1:
            lang_counts["mixed"] += 1

    return gender_counts, lang_counts

def plot_pie(data_dict, title, output_file):
    labels = list(data_dict.keys())
    sizes = list(data_dict.values())
    colors = ['#66b3ff','#ff9999','#99ff99']
    plt.figure(figsize=(6,6))
    print(sizes, labels)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved pie chart to {output_file}")

# === Usage ===
if __name__ == "__main__":
    # Step 1: Set paths
    rttm_folder = "/mmu-audio-ssd/zhenghaorui/others/SD/Large-scale-diarization-dataset/exp/exp1/samples"  # your RTTM folder path
    json_files = [
        "/mmu-audio-ssd/zhenghaorui/others/SD/Large-scale-diarization-dataset/metadata/Chinese/aishell-1/speakers_male.json",
        "/mmu-audio-ssd/zhenghaorui/others/SD/Large-scale-diarization-dataset/metadata/Chinese/aishell-1/speakers_female.json",
        "/mmu-audio-ssd/zhenghaorui/others/SD/Large-scale-diarization-dataset/metadata/English/librispeech/speakers_male.json",
        "/mmu-audio-ssd/zhenghaorui/others/SD/Large-scale-diarization-dataset/metadata/English/librispeech/speakers_female.json"
    ]

    # Step 2: Load speaker info from JSONs
    speaker_info = load_speaker_info(json_files)

    # Step 3: Analyze RTTM files
    gender_stats, lang_stats = analyze_rttm_files(rttm_folder, speaker_info)

    # Step 4: Plot
    plot_pie(gender_stats, "Speaker Gender Composition", "gender_pie.png")
    plot_pie(lang_stats, "Speaker Language Composition", "language_pie.png")
