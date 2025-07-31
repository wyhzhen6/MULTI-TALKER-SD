import os
import argparse
import json
import random



def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def split_data_func(data, ratios):
    total = len(data)
    random.shuffle(data)
    
    split_indices = [int(sum(ratios[:i]) * total) for i in range(1, len(ratios))]
    splits = []
    start = 0
    for end in split_indices:
        splits.append(data[start:end])
        start = end
    splits.append(data[start:])  # the rest goes to the last split
    return splits

def process_file(filename, from_dir, base_dir, subsets, dataset, ratios):
    full_path = os.path.join(from_dir, filename)
    data = load_json(full_path)
    split_results = split_data_func(data, ratios)

    for subset, split_data in zip(subsets, split_results):
        target_dir = os.path.join(base_dir, subset, dataset)
        os.makedirs(target_dir, exist_ok=True)
        out_path = os.path.join(target_dir, filename)
        save_json(split_data, out_path)
        print(f"Saved {subset} data to {out_path} ({len(split_data)} entries)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_dir', required=True, help='Directory containing source JSON files')
    parser.add_argument('--base_dir', required=True, help='Base directory to store subset JSONs')
    parser.add_argument('--dataset', required=True, help='')
    parser.add_argument('--subset', nargs='+', required=True, help='Names of subsets, e.g., train test dev')
    parser.add_argument('--radio', nargs='+', type=float, required=True, help='Ratios for subsets, e.g., 0.8 0.1 0.1')
    parser.add_argument('--seed', default=1234, help='')

    args = parser.parse_args()
    random.seed(args.seed)  # For reproducibility

    if len(args.subset) != len(args.radio):
        raise ValueError("The number of subset names and ratio values must match.")

    total_ratio = sum(args.radio)
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"The sum of radio values must be 1.0, got {total_ratio:.4f}")

    json_files = ['speakers_female.json', 'speakers_male.json']
    for json_file in json_files:
        json_path = os.path.join(args.from_dir, json_file)
        if not os.path.isfile(json_path):
            print(f"Warning: {json_file} not found in {args.from_dir}")
            continue
        process_file(json_file, args.from_dir, args.base_dir, args.subset, args.dataset, args.radio)

if __name__ == '__main__':
    main()
