import os

# Paths to your raw split label directories
label_dirs = [
    r"D:\VSCode\Fine Tune Model\raw_finetune_2_dataset\raw_split\train\labels",
    r"D:\VSCode\Fine Tune Model\raw_finetune_2_dataset\raw_split\valid\labels"
]

print("🔄 Swapping IDs (0 <-> 1) to match VisDrone expectations...")

for target_dir in label_dirs:
    if not os.path.exists(target_dir):
        continue
        
    txt_files = [f for f in os.listdir(target_dir) if f.lower().endswith('.txt')]
    modified_count = 0
    
    for file_name in txt_files:
        file_path = os.path.join(target_dir, file_name)
        
        with open(file_path, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                # Swap logic
                if cls_id == 0:
                    cls_id = 1  # Car becomes 1
                elif cls_id == 1:
                    cls_id = 0  # Human becomes 0
                    
                # Reconstruct the line with the updated class ID
                new_line = f"{cls_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n"
                new_lines.append(new_line)
            else:
                new_lines.append(line)
                
        # Overwrite the file with the corrected labels
        with open(file_path, "w") as f:
            f.writelines(new_lines)
            
        modified_count += 1
        
    print(f"✅ Successfully remapped {modified_count} files in: {os.path.basename(os.path.dirname(target_dir))}")