"""
->> helper script for renaming file names into sanitized well defined structure
"""


from pathlib import Path

def rename_files(base_dir: Path, run_on: list[str] = ['train', 'valid']) -> None:
    for split_name in run_on:
        images_dir: Path = base_dir / split_name / "images"
        labels_dir: Path = base_dir / split_name / "labels"

        if images_dir.exists() and labels_dir.exists():
            # Gather all files
            images: list[Path] = [i for i in images_dir.iterdir() if i.is_file()]
            
            # Using a set for labels speeds up lookups significantly
            labels_set: set[str] = {l.name for l in labels_dir.iterdir() if l.is_file()}

            for idx, img_file in enumerate(images):
                # 1. Map the image name to its expected label name
                expected_label_name = img_file.with_suffix('.txt').name
                
                # 2. Check if the corresponding label file actually exists
                if expected_label_name in labels_set:
                    # Define new names keeping the unique index
                    new_img_name = f"road_detection_{idx}{img_file.suffix}"
                    new_label_name = f"road_detection_{idx}.txt"
                    
                    # Define full target paths
                    new_img_path = img_file.with_name(new_img_name)
                    new_label_path = labels_dir / new_label_name
                    old_label_path = labels_dir / expected_label_name

                    # 3. Rename both files physically on the disk
                    img_file.rename(new_img_path)
                    old_label_path.rename(new_label_path)

            # Re-read directories to print the newly updated names
            updated_images = sorted([i.name for i in images_dir.iterdir() if i.is_file()])
            updated_labels = sorted([l.name for l in labels_dir.iterdir() if l.is_file()])
            
            print(f"--- {split_name} split updated ---")
            print(f"First 2 images: {updated_images[:2]}")
            print(f"First 2 labels: {updated_labels[:2]}\n")

        else:
            print(f"The {split_name} directory structure was not found!")


if __name__ == "__main__":
    dir = Path(r"D:\VSCode\Fine Tune Model\data\raw_split")
    rename_files(dir)

