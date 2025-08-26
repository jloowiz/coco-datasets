import os
import json
import random
import shutil
from pycocotools.coco import COCO

# Paths
src_ann = "annotations/instances_train2017.json"
src_img_dir = "images/train2017"
out_dir = "filtered_12k_coco"
os.makedirs(out_dir, exist_ok=True)

# 1. Load COCO and filter
coco = COCO(src_ann)
classes = ['car','bus','truck','person','motorcycle','bicycle']
cat_ids = coco.getCatIds(catNms=classes)

# Get images that contain ANY of the target categories (not ALL)
all_img_ids = set()
for cat_id in cat_ids:
    img_ids = coco.getImgIds(catIds=[cat_id])
    all_img_ids.update(img_ids)

all_img_ids = list(all_img_ids)
print(f"Found {len(all_img_ids)} images containing target classes")
random.shuffle(all_img_ids)
SELECTED = all_img_ids[:12000]

# Split into train (75%) and validation (25%)
split_point = int(len(SELECTED) * 0.75)
train_img_ids = SELECTED[:split_point]
val_img_ids = SELECTED[split_point:]

print(f"Total selected: {len(SELECTED)} images")
print(f"Training set: {len(train_img_ids)} images (75%)")
print(f"Validation set: {len(val_img_ids)} images (25%)")

# Function to build dataset JSON
def build_dataset(img_ids, dataset_name):
    dataset = {
        "images": [],
        "annotations": [],
        "categories": [cat for cat in coco.dataset['categories'] if cat['name'] in classes]
    }
    ann_id = 0
    for img_id in img_ids:
        img = coco.loadImgs(img_id)[0]
        dataset["images"].append(img)
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)):
            ann["id"] = ann_id
            dataset["annotations"].append(ann)
            ann_id += 1
    
    # Save JSON
    json_file = os.path.join(out_dir, f"instances_{dataset_name}.json")
    with open(json_file, "w") as f:
        json.dump(dataset, f)
    print(f"Saved {dataset_name} annotations: {len(dataset['images'])} images, {len(dataset['annotations'])} annotations")
    
    return dataset

# Build train and validation datasets
train_dataset = build_dataset(train_img_ids, "train")
val_dataset = build_dataset(val_img_ids, "val")

# Create directory structure
train_img_dir = os.path.join(out_dir, "train", "images")
val_img_dir = os.path.join(out_dir, "val", "images")
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)

# Copy training images
print("Copying training images...")
for img in train_dataset["images"]:
    src_path = os.path.join(src_img_dir, img["file_name"])
    dst_path = os.path.join(train_img_dir, img["file_name"])
    shutil.copy(src_path, dst_path)

# Copy validation images
print("Copying validation images...")
for img in val_dataset["images"]:
    src_path = os.path.join(src_img_dir, img["file_name"])
    dst_path = os.path.join(val_img_dir, img["file_name"])
    shutil.copy(src_path, dst_path)

print("Dataset split complete!")
print(f"Output structure:")
print(f"  {out_dir}/")
print(f"    ├── instances_train.json")
print(f"    ├── instances_val.json")
print(f"    ├── train/images/ ({len(train_dataset['images'])} images)")
print(f"    └── val/images/ ({len(val_dataset['images'])} images)")
