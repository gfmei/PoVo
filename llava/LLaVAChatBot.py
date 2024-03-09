import requests
from PIL import Image

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, pipeline
from transformers.models.llava import LlavaForConditionalGeneration, LlavaPreTrainedModel

model_id = "llava-hf/llava-1.5-13b-hf"

prompt = "USER: <image>\nList all objects in this image?\nASSISTANT:"
image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"

# model = LlavaForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True,
#     # load_in_4bit=True
# ).to(0)
#
# raw_image = Image.open(requests.get(image_file, stream=True).raw)
# inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
#
# output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
# print(processor.decode(output[0][2:], skip_special_tokens=True))
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})
max_new_tokens = 200
image = Image.open(requests.get(image_file, stream=True).raw)
outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)
print(outputs[0]["generated_text"])
processor = AutoProcessor.from_pretrained(model_id)

