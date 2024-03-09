# This file contains the HEAD - COMMON - TAIL split category ids for ScanNet 200

HEAD_CATS_SCANNET_200 = ['tv stand', 'curtain', 'blinds', 'shower curtain', 'bookshelf', 'tv', 'kitchen cabinet',
                         'pillow', 'lamp', 'dresser', 'monitor', 'object', 'ceiling', 'board', 'stove', 'closet wall',
                         'couch', 'office chair', 'kitchen counter', 'shower', 'closet', 'doorframe', 'sofa chair',
                         'mailbox', 'nightstand', 'washing machine', 'picture', 'book', 'sink', 'recycling bin',
                         'table', 'backpack', 'shower wall', 'toilet', 'copier', 'counter', 'stool', 'refrigerator',
                         'window', 'file cabinet', 'chair', 'wall', 'plant', 'coffee table', 'stairs', 'armchair',
                         'cabinet', 'bathroom vanity', 'bathroom stall', 'mirror', 'blackboard', 'trash can',
                         'stair rail', 'box', 'towel', 'door', 'clothes', 'whiteboard', 'bed', 'floor', 'bathtub',
                         'desk', 'wardrobe', 'clothes dryer', 'radiator', 'shelf']
COMMON_CATS_SCANNET_200 = ["cushion", "end table", "dining table", "keyboard", "bag", "toilet paper", "printer",
                           "blanket", "microwave", "shoe", "computer tower", "bottle", "bin", "ottoman", "bench",
                           "basket", "fan", "laptop", "person", "paper towel dispenser", "oven", "rack", "piano",
                           "suitcase", "rail", "container", "telephone", "stand", "light", "laundry basket", "pipe",
                           "seat", "column", "bicycle", "ladder", "jacket", "storage bin", "coffee maker", "dishwasher",
                           "machine", "mat", "windowsill", "bulletin board", "fireplace", "mini fridge", "water cooler",
                           "shower door", "pillar", "ledge", "furniture", "cart", "decoration", "closet door",
                           "vacuum cleaner", "dish rack", "range hood", "projector screen", "divider",
                           "bathroom counter", "laundry hamper", "bathroom stall door", "ceiling light", "trash bin",
                           "bathroom cabinet", "structure", "storage organizer", "potted plant", "mattress"]
TAIL_CATS_SCANNET_200 = ["paper", "plate", "soap dispenser", "bucket", "clock", "guitar", "toilet paper holder",
                         "speaker", "cup", "paper towel roll", "bar", "toaster", "ironing board", "soap dish",
                         "toilet paper dispenser", "fire extinguisher", "ball", "hat", "shower curtain rod",
                         "paper cutter", "tray", "toaster oven", "mouse", "toilet seat cover dispenser",
                         "storage container", "scale", "tissue box", "light switch", "crate", "power outlet", "sign",
                         "projector", "candle", "plunger", "stuffed animal", "headphones", "broom", "guitar case",
                         "dustpan", "hair dryer", "water bottle", "handicap bar", "purse", "vent", "shower floor",
                         "water pitcher", "bowl", "paper bag", "alarm clock", "music stand", "laundry detergent",
                         "dumbbell", "tube", "cd case", "closet rod", "coffee kettle", "shower head", "keyboard piano",
                         "case of water bottles", "coat rack", "folded chair", "fire alarm", "power strip", "calendar",
                         "poster", "luggage"]

# Given the different size of the official train and val sets, not all ScanNet200 categories are present in the validation set.
# Here we list of categories with labels and IDs present in both train and validation set, and the remaining categories those are present in train, but not in val
# We dont evaluate on unseen validation categories in this benchmark

VALID_CLASS_IDS_200_VALIDATION = (
'wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink',
'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box',
'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool',
'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag', 'backpack', 'toilet paper',
'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave', 'stove',
'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier', 'basket',
'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 'paper towel dispenser', 'oven', 'blinds',
'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container', 'wardrobe',
'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar',
'toilet paper holder', 'seat', 'speaker', 'column', 'ladder', 'bathroom stall', 'shower wall', 'cup', 'jacket',
'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat', 'windowsill', 'bar', 'toaster',
'bulletin board', 'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe', 'toilet paper dispenser',
'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray',
'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser', 'furniture', 'cart', 'scale',
'tissue box', 'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door',
'vacuum cleaner', 'plunger', 'stuffed animal', 'headphones', 'dish rack', 'broom', 'range hood', 'dustpan',
'hair dryer', 'water bottle', 'handicap bar', 'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag',
'projector screen', 'divider', 'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall',
'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube',
'bathroom cabinet', 'closet rod', 'coffee kettle', 'shower head', 'keyboard piano', 'case of water bottles',
'coat rack', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'mattress')

CLASS_LABELS_200_VALIDATION = (
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36,
38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
106, 107, 110, 112, 115, 116, 118, 120, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154, 155,
156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 229, 230, 232,
233, 242, 250, 261, 264, 276, 283, 300, 304, 312, 323, 325, 342, 356, 370, 392, 395, 408, 417, 488, 540, 562, 570, 609,
748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1175, 1176, 1179, 1180, 1181, 1182,
1184, 1185, 1186, 1187, 1188, 1189, 1191)

VALID_CLASS_IDS_200_TRAIN_ONLY = (
'bicycle', 'storage container', 'candle', 'guitar case', 'purse', 'alarm clock', 'music stand', 'cd case', 'structure',
'storage organizer', 'luggage')

CLASS_LABELS_200_TRAIN_ONLY = (121, 221, 286, 331, 399, 572, 581, 1174, 1178, 1183, 1190)


def get_bidmaps_id_cat(label_ids, label_cats):
    # Constructing the cat_ids dictionary (category name -> ID)
    cat_ids = {cat: id for id, cat in zip(label_ids, label_cats)}

    # Constructing the id_cats dictionary (ID -> category name)
    id_cats = {id: cat for id, cat in zip(label_ids, label_cats)}

    return cat_ids, id_cats


CAT2ID, ID2CAT = get_bidmaps_id_cat(CLASS_LABELS_200_VALIDATION, VALID_CLASS_IDS_200_VALIDATION)



if __name__ == '__main__':
    cids, idcs = get_bidmaps_id_cat(CLASS_LABELS_200_VALIDATION, VALID_CLASS_IDS_200_VALIDATION)
    print(cids, idcs)
