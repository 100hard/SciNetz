---
license: apache-2.0
---

THIS IS WORK IN PROGRESS


# Docling Layout Model

`docling-layout-heron` is the Layout Model of [Docling project](https://github.com/docling-project/docling).

This model uses the [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch) architecture and has been trained from scratch on a variety of document datasets.


# Inference code example

Prerequisites:

```bash
pip install transformers Pillow torch requests
```

Prediction:

```python
import requests
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
import torch
from PIL import Image


classes_map = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
    11: "Document Index",
    12: "Code",
    13: "Checkbox-Selected",
    14: "Checkbox-Unselected",
    15: "Form",
    16: "Key-Value Region",
}
image_url = "https://huggingface.co/spaces/ds4sd/SmolDocling-256M-Demo/resolve/main/example_images/annual_rep_14.png"
model_name = "ds4sd/docling-layout-heron"
threshold = 0.6


# Download the image
image = Image.open(requests.get(image_url, stream=True).raw)
image = image.convert("RGB")

# Initialize the model
image_processor = RTDetrImageProcessor.from_pretrained(model_name)
model = RTDetrV2ForObjectDetection.from_pretrained(model_name)

# Run the prediction pipeline
inputs = image_processor(images=[image], return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
results = image_processor.post_process_object_detection(
    outputs,
    target_sizes=torch.tensor([image.size[::-1]]),
    threshold=threshold,
)

# Get the results
for result in results:
    for score, label_id, box in zip(
        result["scores"], result["labels"], result["boxes"]
    ):
        score = round(score.item(), 2)
        label = classes_map[label_id.item()]
        box = [round(i, 2) for i in box.tolist()]
        print(f"{label}:{score} {box}")
```


# References

```
@techreport{Docling,
  author = {Deep Search Team},
  month = {8},
  title = {Docling Technical Report},
  url = {https://arxiv.org/abs/2408.09869v4},
  eprint = {2408.09869},
  doi = {10.48550/arXiv.2408.09869},
  version = {1.0.0},
  year = {2024}
}

@misc{lv2024rtdetrv2improvedbaselinebagoffreebies,
      title={RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer},
      author={Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
      year={2024},
      eprint={2407.17140},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.17140},
}

```