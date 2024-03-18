import clip
import numpy as np
import torch
from PIL import Image
from skimage.color import label2rgb
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

from libs.lib_mask import get_masks_from_pil

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# Function to preprocess images for CLIP in batch
def clip_preprocess_batch(images):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    return torch.stack([transform(image) for image in images])


# Load your image
image_path = 'test1.jpg'
image = Image.open(image_path)
image_np = np.array(image)

# Generate superpixels
segments, superpixel_images = get_masks_from_pil(image, n_seg=16)
# Preprocess all superpixel images at once
preprocessed_batch = clip_preprocess_batch(superpixel_images).to(device)

# Extract features in a batch
with torch.no_grad():
    batch_features = model.encode_image(preprocessed_batch)

# Assuming batch_features holds the CLIP-encoded features for each superpixel

# Example class descriptions
class_descriptions = ['a photo of a bag', 'a photo of a desk', 'a photo of a chair']
text_tokens = clip.tokenize(class_descriptions).to(device)

# Get text features for the descriptions
with torch.no_grad():
    text_features = model.encode_text(text_tokens)

# Calculate similarity (here using cosine similarity)
print(batch_features.shape, segments.shape)
similarities = torch.matmul(batch_features, text_features.T)
similarities = torch.softmax(similarities, dim=-1)

# Classify each superpixel by highest similarity score
classification_results = torch.argmax(similarities, dim=1).cpu().numpy()

# Recompose the segmented image based on classification results
segmented_image = np.zeros(segments.shape, dtype=np.int32)
for segment_id, class_id in enumerate(np.unique(segments)):
    segmented_image[segments == segment_id] = classification_results[segment_id]

# Optionally, visualize the segmentation
# Map classification results to colors
segmented_image_rgb = label2rgb(segmented_image, image_np, bg_label=-1, kind='avg')


