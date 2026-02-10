import os
import argparse
from collections import defaultdict

# 1. Criteria from Section 3.1 of the paper
TARGET_GENRES = ['gBR', 'gPO', 'gLO', 'gWA', 'gMH', 'gLH', 'gHO', 'gKR', 'gJS', 'gJB']
TARGET_SITUATIONS = ['sBM', 'sFM'] # Basic and Advanced
TARGET_CAMERA = 'c01'               # Frontal view
TARGET_PER_GENRE = 60               # 10 genres * 60 = 600 total

def filter_aist_videos(all_files):
    genre_map = defaultdict(list)
    
    # Sort files to ensure deterministic selection
    for filename in sorted(all_files):
        # Format: {Genre}_{Situation}_{Camera}_{Dancer}_{Music}_{Choreography}.mp4
        parts = filename.split('_')
        if len(parts) < 3: continue
        
        genre, situation, camera = parts[0], parts[1], parts[2]
        
        if (genre in TARGET_GENRES and 
            situation in TARGET_SITUATIONS and 
            camera == TARGET_CAMERA):
            genre_map[genre].append(filename)

    original_count = sum(len(v) for v in genre_map.values())
    print(f"Total valid candidates found: {original_count}")

    final_600_list = []
    for g in TARGET_GENRES:
        subset = genre_map[g][:TARGET_PER_GENRE]
        final_600_list.extend(subset)
        
        if len(subset) < TARGET_PER_GENRE:
            print(f"Warning: Genre {g} only has {len(subset)} videos.")

    return final_600_list

def main():
    # Set up argparse
    parser = argparse.ArgumentParser(description="Filter AIST++ dataset videos based on specific criteria.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing AIST videos")
    parser.add_argument("--output", type=str, default="aist_filtered_videos.txt", help="Name of the output text file")
    
    args = parser.parse_args()

    # Check if directory exists
    if not os.path.isdir(args.data_dir):
        print(f"Error: The directory '{args.data_dir}' does not exist.")
        return

    # Process files
    all_files = os.listdir(args.data_dir)
    target_videos = filter_aist_videos(all_files)
    
    # Save the filtered list to a text file
    with open(args.output, 'w') as f:
        for video in target_videos:
            f.write(f"{video}\n")
            
    print(f"Successfully saved {len(target_videos)} video filenames to {args.output}")

if __name__ == "__main__":
    main()