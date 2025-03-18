import nltk
import requests
import spacy, re
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
import torch
from textblob import TextBlob
# import inflection

# Load the spaCy model
import inflect
p_eg = inflect.engine()
nlp = spacy.load("en_core_web_lg")
nltk.download('averaged_perceptron_tagger')


# p_singular = inflect.engine()

def remove_numbers_and_alphabet(sentence):
    # This regex will match any numbers and any alphabetic characters
    pattern = '[0-9a-zA-Z]'
    # Substitute all occurrences of the pattern with an empty string
    return re.sub(pattern, '', sentence)


def filter_list(elements):
    filtered_elements = [
        element for element in elements
        if element is not None
           and element.strip() != ''
           and not element.strip().isdigit()
           and len(element.strip()) > 1  # Exclude single character strings
    ]
    return filtered_elements or ['floor']


def remove_special_signs_and_numbers(items):
    cleaned_items = []
    for item in items:
        # Ensure item is a string
        if not isinstance(item, str):
            continue

        # Remove special characters and numbers except alphanumeric and spaces, replace with space
        cleaned_item = re.sub(r'[^a-zA-Z0-9\s]+', '', item)
        cleaned_item = cleaned_item.strip()

        if not cleaned_item:  # If the string is empty after cleaning
            cleaned_items.append('floor')
            continue

        # If more than three words, perform entity recognition on the original item
        if len(cleaned_item.split()) > 2:
            doc = nlp(item)
            # Extract text of each entity and use these if available
            if entity_words := [ent.text for ent in doc.ents]:
                cleaned_items.extend(entity_words)
            else:
                cleaned_items.append('floor')  # Append 'floor' if no entities are extracted
        else:
            # Use the cleaned item if less than or equal to three words
            cleaned_items.append(cleaned_item)

    return cleaned_items


def remove_specific_words(word_list):
    words_to_remove = {'background', 'box', 'contours'}
    return [word for word in word_list if word not in words_to_remove]


def extract_noun_phrases(sentence):
    if '[/INST]' in sentence:
        txt = sentence.split('[/INST]')[-1].strip().lower()
    else:
        txt = sentence
    labels = ['floor']
    # labels = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(txt.strip().lower())) if pos[0] == 'N']
    try:
        labels.extend([word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(txt.strip().lower())) if pos[0] == 'N'])
    except Exception:
        labels.append('wall')
    #############################################################################
    # try:
    #     gfg = TextBlob(txt)
    #     # nouns = [word.singularize() for word, tag in gfg.tags if tag.startswith('NN')]
    #     # print(nouns)
    #     labels.extend(gfg.noun_phrases)
    # except Exception:
    #     pass
    labels = [p_eg.singular_noun(word_) for word_ in labels]
    print(labels)
    return labels


def load_llava_model_hg(model_path="llava-hf/llava-v1.6-mistral-7b-hf", device='cuda'):
    processor = AutoProcessor.from_pretrained(model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto')
    processor.tokenizer.padding_side = "left"
    # tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
    model.to(device)

    return model, processor


def get_image_category_from_llavahg(images, model, processor, prompt, device="cuda:0"):
    # Model
    if prompt is None:
        prompt = ["[INST] <image>\nWhat objects and background is shown in this image? [/INST]",]
    # Generate
    inputs = processor(text=prompt * len(images), images=images, padding=True, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, max_new_tokens=500)
    # outputs = processor.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    labels = []
    for ids in generate_ids:
        output = processor.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        label = extract_noun_phrases(output)
        labels.append(list(set(label)))
    return labels


if __name__ == '__main__':
    url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png"
    image = Image.open(requests.get(url, stream=True).raw)
    prompt = ['[INST] <image>List all foreground and background in the image? [/INST]']

    # inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    model, processor = load_llava_model_hg(model_path="llava-hf/llava-v1.6-mistral-7b-hf", device='cuda')
    images = [image, Image.open('test1.jpg')]
    lbs = get_image_category_from_llavahg(images, model, processor, prompt, device="cuda:0")
    print(lbs)
