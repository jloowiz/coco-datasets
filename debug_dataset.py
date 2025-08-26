import os
import json
from pycocotools.coco import COCO

# Paths
src_ann = "annotations/instances_train2017.json"
src_img_dir = "images/train2017"

print("Loading COCO annotation file...")
coco = COCO(src_ann)

# Get target classes
classes = ['car','bus','truck','person','motorcycle','bicycle']
cat_ids = coco.getCatIds(catNms=classes)
print(f"Found category IDs: {cat_ids}")
print(f"Categories: {[(cat_id, coco.loadCats(cat_id)[0]['name']) for cat_id in cat_ids]}")

# Get all images with these categories
all_img_ids = coco.getImgIds(catIds=cat_ids)
print(f"Total images with target classes: {len(all_img_ids)}")

# Check how many of these images actually exist in the directory
existing_images = []
missing_images = []

print("Checking first 50 images for existence...")
for i, img_id in enumerate(all_img_ids[:50]):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(src_img_dir, img_info["file_name"])
    if os.path.exists(img_path):
        existing_images.append(img_id)
    else:
        missing_images.append(img_id)
    
    if i < 10:  # Show first 10 for debugging
        print(f"  Image {img_id}: {img_info['file_name']} - {'EXISTS' if os.path.exists(img_path) else 'MISSING'}")

print(f"\nIn first 50 images:")
print(f"  Existing: {len(existing_images)}")
print(f"  Missing: {len(missing_images)}")

# Check a few image files in the directory
print(f"\nSample images in {src_img_dir}:")
img_files = os.listdir(src_img_dir)[:10]
for img_file in img_files:
    print(f"  {img_file}")
