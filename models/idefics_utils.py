import nltk
import requests
import torch
from nltk.tokenize import word_tokenize
from transformers import AutoModelForVision2Seq
from transformers import BitsAndBytesConfig, AutoProcessor

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)


def filter_list(elements):
    filtered_elements = [
        element for element in elements
        if element is not None
           and element.strip() != ''
           and not element.strip().isdigit()
           and len(element.strip()) > 1  # Exclude single character strings
    ]
    return filtered_elements or ['floor']


def extract_nouns(sentence):
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    # Get part-of-speech tags for each token
    tagged = nltk.pos_tag(tokens)
    # Extract words tagged as various forms of nouns (NN for singular nouns, NNS for plural, NNP for proper singular,
    # NNPS for proper plural)
    nouns = [word for word, pos in tagged if pos in ("NN", "NNS", "NNP", "NNPS")]
    return nouns


def load_idefics_model(model_path="HuggingFaceM4/idefics2-8b", device='cuda:0'):
    """Load Llava"""

    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        quantization_config=quantization_config
    )
    model.eval()
    processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")

    return model, processor


def get_image_category_from_idefics(images, model, processor, messages=None, device='cuda'):
    if messages is None:
        messages = "List all objects and background of this image?"
    prompts = [messages] * len(images)
    # text = processor.apply_chat_template(prompts, add_generation_prompt=True)
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(outputs)
    labels = []
    for output in outputs:
        output = output.strip().lower()  # Normalize the casing and strip whitespace
        label = []
        # Check if output is empty or contains specific phrases indicating no valid objects
        if not output:
            label.append('floor')
        else:
            label.extend(extract_nouns(output) or ['floor'])
        # Ensure no label is an empty string, replace empty labels with 'floor'
        label = filter_list(label or ['floor'])
        labels.extend(label)
    return set(labels)


if __name__ == '__main__':
    from PIL import Image

    model_path = "HuggingFaceM4/idefics2-8b"
    model, processor = load_idefics_model(model_path)

    url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
    url_2 = "http://images.cocodataset.org/val2017/000000219578.jpg"

    image_1 = Image.open(requests.get(url_1, stream=True).raw)
    image_2 = Image.open(requests.get(url_2, stream=True).raw)
    images = [[image_1], [image_2],]

    labels = get_image_category_from_idefics(images, model, processor, messages=None, device='cuda')
    print(labels)
