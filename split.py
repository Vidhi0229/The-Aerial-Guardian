import os
import shutil
import random

sequences_root = "sequences"
labels_root = "labels"   
output_base = "dataset"

train_ratio = 0.80

sequences = sorted(os.listdir(sequences_root))
random.shuffle(sequences)

split_index = int(len(sequences) * train_ratio)

train_seqs = sequences[:split_index]
val_seqs = sequences[split_index:]

print(f"Total sequences: {len(sequences)}")
print(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}")
print(f"Train seqs: {train_seqs}")
print(f"Val seqs: {val_seqs}")

for split in ["train", "val"]:
    os.makedirs(os.path.join(output_base, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_base, "labels", split), exist_ok=True)

def process_sequence(seq_name, split):
    seq_path = os.path.join(sequences_root, seq_name)

    copied = 0
    label_found = 0
    label_missing = 0

    for img_name in sorted(os.listdir(seq_path)):
        if not img_name.endswith(".jpg"):
            continue

        img_src = os.path.join(seq_path, img_name)

        label_name = img_name.replace(".jpg", ".txt")
        label_src = os.path.join(labels_root, f"{seq_name}_{label_name}")

        new_name = f"{seq_name}_{img_name}"

        img_dst = os.path.join(output_base, "images", split, new_name)
        label_dst = os.path.join(output_base, "labels", split, new_name.replace(".jpg", ".txt"))

        shutil.copy(img_src, img_dst)
        copied += 1

        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)
            label_found += 1
        else:
            open(label_dst, "w").close()
            label_missing += 1
            if label_missing <= 2:
                print(f"  [MISSING LABEL] expected: {label_src}")

    print(f"  [{split}] {seq_name}: {copied} images, {label_found} labels found, {label_missing} missing")

for seq in train_seqs:
    process_sequence(seq, "train")

for seq in val_seqs:
    process_sequence(seq, "val")

print("\ncompleted")

for split in ["train", "val"]:
    imgs = len(os.listdir(os.path.join(output_base, "images", split)))
    labs = len([f for f in os.listdir(os.path.join(output_base, "labels", split))
                if os.path.getsize(os.path.join(output_base, "labels", split, f)) > 0])
    empty = len([f for f in os.listdir(os.path.join(output_base, "labels", split))
                 if os.path.getsize(os.path.join(output_base, "labels", split, f)) == 0])
    print(f"{split}: {imgs} images | {labs} non-empty labels | {empty} empty labels")