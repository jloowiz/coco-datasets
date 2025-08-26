from pycocotools.coco import COCO

# Load COCO
coco = COCO("annotations/instances_train2017.json")

# Get all categories
all_cats = coco.loadCats(coco.getCatIds())
print("All available categories:")
for cat in all_cats:
    print(f"  ID {cat['id']}: {cat['name']}")

print("\n" + "="*50)

# Check our target classes
classes = ['car','bus','truck','person','motorcycle','bicycle']
print(f"\nLooking for classes: {classes}")

for class_name in classes:
    cat_ids = coco.getCatIds(catNms=[class_name])
    if cat_ids:
        img_ids = coco.getImgIds(catIds=cat_ids)
        print(f"  {class_name}: Category ID {cat_ids[0]}, {len(img_ids)} images")
    else:
        print(f"  {class_name}: NOT FOUND!")

# Try alternative names
print(f"\nTrying alternative category names:")
alt_names = ['automobile', 'vehicle', 'motorbike', 'bike', 'pedestrian']
for alt_name in alt_names:
    cat_ids = coco.getCatIds(catNms=[alt_name])
    if cat_ids:
        img_ids = coco.getImgIds(catIds=cat_ids)
        print(f"  {alt_name}: Category ID {cat_ids[0]}, {len(img_ids)} images")
