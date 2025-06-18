import os
import json
import matplotlib.pyplot as plt

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

def classify_combination(genders, langs):
    # Gender classification
    if len(genders) == 1:
        gender_class = list(genders)[0]
    else:
        gender_class = 'mixed'

    # Language classification
    if len(langs) == 1:
        lang_class = list(langs)[0]
    else:
        lang_class = 'mixed'

    return lang_class, gender_class

def analyze_rttm_nested(rttm_folder, speaker_info):
    nested_counts = {
        'chinese': {'male': 0, 'female': 0, 'mixed': 0},
        'english': {'male': 0, 'female': 0, 'mixed': 0},
        'mixed':   {'male': 0, 'female': 0, 'mixed': 0}
    }

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

        lang_class, gender_class = classify_combination(genders, langs)
        nested_counts[lang_class][gender_class] += 1

    return nested_counts

def plot_nested_pie(nested_counts, output_file="sunburst_language_gender.png"):
    outer_labels = []
    outer_sizes = []

    for lang, gender_dict in nested_counts.items():
        total = sum(gender_dict.values())
        outer_labels.append(lang)
        outer_sizes.append(total)

    # Inner ring: gender composition within each language group
    inner_labels = []
    inner_sizes = []
    inner_colors = []

    color_map = {
        'chinese': {'male': '#ADD8E6', 'female': '#FFB6C1', 'mixed': '#90EE90'},
        'english': {'male': '#87CEEB', 'female': '#FF69B4', 'mixed': '#32CD32'},
        'mixed':   {'male': '#6495ED', 'female': '#FF1493', 'mixed': '#228B22'},
    }

    for lang in nested_counts:
        for gender in ['male', 'female', 'mixed']:
            count = nested_counts[lang].get(gender, 0)
            if count > 0:
                inner_labels.append(f"{lang}-{gender}")
                inner_sizes.append(count)
                inner_colors.append(color_map[lang][gender])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(outer_sizes, radius=1, labels=outer_labels, labeldistance=0.8,
           wedgeprops=dict(width=0.3, edgecolor='w'))
    ax.pie(inner_sizes, radius=0.7, labels=inner_labels, labeldistance=0.9,
           colors=inner_colors, wedgeprops=dict(width=0.3, edgecolor='w'))

    plt.title("RTTM Composition by Language and Gender")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    print(f"Saved nested pie chart to: {output_file}")

# === Main execution ===
if __name__ == "__main__":
    # TODO: Replace with your actual paths
    rttm_folder = "/mmu-audio-ssd/zhenghaorui/others/SD/Large-scale-diarization-dataset/exp/exp1/samples"  # your RTTM folder path
    json_files = [
        "/mmu-audio-ssd/zhenghaorui/others/SD/Large-scale-diarization-dataset/metadata/Chinese/aishell-1/speakers_male.json",
        "/mmu-audio-ssd/zhenghaorui/others/SD/Large-scale-diarization-dataset/metadata/Chinese/aishell-1/speakers_female.json",
        "/mmu-audio-ssd/zhenghaorui/others/SD/Large-scale-diarization-dataset/metadata/English/librispeech/speakers_male.json",
        "/mmu-audio-ssd/zhenghaorui/others/SD/Large-scale-diarization-dataset/metadata/English/librispeech/speakers_female.json"
    ]

    speaker_info = load_speaker_info(json_files)
    nested_counts = analyze_rttm_nested(rttm_folder, speaker_info)
    plot_nested_pie(nested_counts)
