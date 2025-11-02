<div align="center">
  <h1>OCRVerse: Towards Holistic OCR in End-to-End  Vision-Language Models</h1>
</div>

<div align="center">
<!-- <a href=''><img src='https://img.shields.io/badge/Arxiv-2507.15509-b31b1b.svg?logo=arXiv'></a>&ensp;
<a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face%20-models-blue'></a>&ensp; -->
<a href=https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE><img src='https://img.shields.io/badge/License-Apache_2.0-green.svg'></a>

<!-- Lei Chen, Xuanle Zhao, Zhixiong Zeng†, Jing Huang, Yufeng Zhong, Lin Ma* -->
</div>

<!-- <div align="center">
<strong>Meituan Group</strong>
</div>
<div align="center">
† Project Leader; * Corresponding Author
</div> -->

---

<!-- **Chart-R1** is a vision-language model that enables complex chart reasoning through reinforcement learning fine-tuning. As the **first** to apply R1-Style methods to the chart domain, it employs programmatic data synthesis to generate high-quality step-by-step reasoning data for charts. Chart-R1's two-stage training includes Chart-COT (chain-of-thought supervision) and Chart-RFT (numerically sensitive reinforcement fine-tuning). Experiments show Chart-R1 achieves significant advantages on open-source benchmarks and the ChartRQA dataset, comparable to large-scale models like GPT-4o and Claude-3.5, proving R1-Style effectiveness for chart reasoning.
<div align="center">
<img src="./assets/chart_r1_radar.png"  width="100%">
</div> -->

# 📢 News and Updates
* ```2025.10.27``` We upload our model weights [OCRVerse-text](https://huggingface.co/DocTron/OCRVerse-text) to HuggingFace.
<!-- * ```2025.07.21``` 🔥🔥🔥 We release the technical report of **Chart-R1** at arXiv [link](https://arxiv.org/abs/2507.15509). -->


# 🤗 Models
|  Model   | Download Link  |
|  ----  | ----  |
|  OCRVerse-text |  [DocTron/OCRVerse-text](https://huggingface.co/DocTron/OCRVerse-text)  |
<!-- |  OCRVerse-code  |  [DocTron/OCRVerse-code](https://huggingface.co/DocTron/OCRVerse-code)   | -->

<!-- The ```Chart-COT``` is Qwen2.5-VL-7B-Instruct fine-tuned with supervised learning on the ChartRQA-SFT dataset. The ```Chart-R1``` is Chart-COT further optimized through reinforcement fine-tuning (RFT). -->

# 📥 Data Processing

To build a multi-scenario, multi-type document OCR dataset, we combine open-source and self-built data to balance scale and quality:
- Open-source data is low-cost and large-scale but suffers from uneven quality due to scattered sources and lack of unified annotation standards. We use VLM for quality optimization to improve usability.
- To cover gaps in real-world scenarios, self-built data serves as a key supplement:
  - Collect real PDF documents (matching practical layouts, fonts, colors, and resolutions) with VLM-powered precise annotation.
  - Crawl public high-quality online documents and convert them to images via browser rendering to enrich data types and expand scenario coverage.

![数据处理流程图](assets/数据处理流程.png)

# 📊 Performance

## OmniDocBench v1.5

### End-to-End Evaluation

End-to-end evaluation assesses the model's accuracy in parsing PDF page content. The evaluation uses the model's Markdown output of the entire PDF page parsing results as the prediction. The Overall metric is calculated as:

$$
\text{Overall} = \frac{(1-\text{Text Edit Distance}) \times 100 + \text{Table TEDS} +\text{Formula CDM}}{3}
$$

<table>
  <thead>
    <tr>
      <th>Model Type</th>
      <th>Methods</th>
      <th>Release Date</th>
      <th>End to End</th>
      <th>Parameters</th>
      <th>Overall↑</th>
      <th>Text<sup>Edit</sup>↓</th>
      <th>Formula<sup>CDM</sup>↑</th>
      <th>Table<sup>TEDS</sup>↑</th>
      <th>Table<sup>TEDS-S</sup>↑</th>
      <th>Reading Order<sup>Edit</sup>↓</th>
    </tr>
  </thead>
  <tbody>
    <!-- Pipeline Tools -->
    <tr>
      <td rowspan="3">Pipeline Tools</td>
      <td>Marker-1.8.2</td>
      <td>2025</td>
      <td>❌</td>
      <td>-</td>
      <td>71.30</td>
      <td>0.206</td>
      <td>76.66</td>
      <td>57.88</td>
      <td>71.17</td>
      <td>0.250</td>
    </tr>
    <tr>
      <td>Mineru2-pipeline</td>
      <td>2025</td>
      <td>❌</td>
      <td>-</td>
      <td>75.51</td>
      <td>0.209</td>
      <td>76.55</td>
      <td>70.90</td>
      <td>79.11</td>
      <td>0.225</td>
    </tr>
    <tr>
      <td>PP-StructureV3</td>
      <td>2024</td>
      <td>❌</td>
      <td>-</td>
      <td>86.73</td>
      <td>0.073</td>
      <td>85.79</td>
      <td>81.68</td>
      <td>89.48</td>
      <td>0.073</td>
    </tr>
    <!-- General VLMs -->
    <tr>
      <td rowspan="5">General VLMs</td>
      <td>GPT-4o</td>
      <td>2024</td>
      <td>✅</td>
      <td>-</td>
      <td>75.02</td>
      <td>0.217</td>
      <td>79.70</td>
      <td>67.07</td>
      <td>76.09</td>
      <td>0.148</td>
    </tr>
    <tr>
      <td>InternVL3-76B</td>
      <td>2025</td>
      <td>✅</td>
      <td>76B</td>
      <td>80.33</td>
      <td>0.131</td>
      <td>83.42</td>
      <td>70.64</td>
      <td>77.74</td>
      <td>0.113</td>
    </tr>
    <tr>
      <td>InternVL3.5-241B</td>
      <td>2025</td>
      <td>✅</td>
      <td>241B</td>
      <td>82.67</td>
      <td>0.142</td>
      <td>87.23</td>
      <td>75.00</td>
      <td>81.28</td>
      <td>0.125</td>
    </tr>
    <tr>
      <td>Qwen2.5-VL-72B</td>
      <td>2025</td>
      <td>✅</td>
      <td>72B</td>
      <td>87.02</td>
      <td>0.094</td>
      <td>88.27</td>
      <td>82.15</td>
      <td>86.22</td>
      <td>0.102</td>
    </tr>
    <tr>
      <td>Gemini-2.5 Pro</td>
      <td>2025</td>
      <td>✅</td>
      <td>-</td>
      <td>88.03</td>
      <td>0.075</td>
      <td>85.82</td>
      <td>85.71</td>
      <td>90.29</td>
      <td>0.097</td>
    </tr>
    <!-- Specialized VLMs: End to End ❌ -->
    <tr>
      <td rowspan="15">Specialized VLMs</td>
      <td>Dolphin</td>
      <td>2025.05</td>
      <td>❌</td>
      <td>322M</td>
      <td>74.67</td>
      <td>0.125</td>
      <td>67.85</td>
      <td>68.70</td>
      <td>77.77</td>
      <td>0.124</td>
    </tr>
    <tr>
      <td>MinerU2-VLM</td>
      <td>2025.06</td>
      <td>❌</td>
      <td>0.9B</td>
      <td>85.56</td>
      <td>0.078</td>
      <td>80.95</td>
      <td>83.54</td>
      <td>87.66</td>
      <td>0.086</td>
    </tr>
    <tr>
      <td>MonkeyOCR-pro-1.2B</td>
      <td>2025.07</td>
      <td>❌</td>
      <td>1.9B</td>
      <td>86.96</td>
      <td>0.084</td>
      <td>85.02</td>
      <td>84.24</td>
      <td>89.02</td>
      <td>0.130</td>
    </tr>
    <tr>
      <td>MonkeyOCR-3B</td>
      <td>2025.06</td>
      <td>❌</td>
      <td>3.7B</td>
      <td>87.13</td>
      <td>0.075</td>
      <td>87.45</td>
      <td>81.39</td>
      <td>85.92</td>
      <td>0.129</td>
    </tr>
    <tr>
      <td>MonkeyOCR-pro-3B</td>
      <td>2025.07</td>
      <td>❌</td>
      <td>3.7B</td>
      <td>88.85</td>
      <td>0.075</td>
      <td>87.25</td>
      <td>86.78</td>
      <td>90.63</td>
      <td>0.128</td>
    </tr>
    <tr>
      <td>MinerU2.5</td>
      <td>2025.09</td>
      <td>❌</td>
      <td>1.2B</td>
      <td>90.67</td>
      <td>0.047</td>
      <td>88.46</td>
      <td>88.22</td>
      <td>92.38</td>
      <td>0.044</td>
    </tr>
    <tr>
      <td>PaddleOCR-VL</td>
      <td>2025.10</td>
      <td>❌</td>
      <td>0.9B</td>
      <td>92.56</td>
      <td>0.035</td>
      <td>91.43</td>
      <td>89.76</td>
      <td>93.52</td>
      <td>0.043</td>
    </tr>
    <!-- Specialized VLMs: End to End ✅ -->
    <tr>
      <td>OCRFlux-3B</td>
      <td>2025.06</td>
      <td>✅</td>
      <td>3B</td>
      <td>74.82</td>
      <td>0.193</td>
      <td>68.03</td>
      <td>75.75</td>
      <td>80.23</td>
      <td>0.202</td>
    </tr>
    <tr>
      <td>Mistral OCR</td>
      <td>2025.03</td>
      <td>✅</td>
      <td>-</td>
      <td>78.83</td>
      <td>0.164</td>
      <td>82.84</td>
      <td>70.03</td>
      <td>78.04</td>
      <td>0.144</td>
    </tr>
    <tr>
      <td>POINTS-Reader</td>
      <td>2025.08</td>
      <td>✅</td>
      <td>3B</td>
      <td>80.98</td>
      <td>0.134</td>
      <td>79.20</td>
      <td>77.13</td>
      <td>81.66</td>
      <td>0.145</td>
    </tr>
    <tr>
      <td>olmOCR-7B</td>
      <td>2025.02</td>
      <td>✅</td>
      <td>7B</td>
      <td>81.79</td>
      <td>0.096</td>
      <td>86.04</td>
      <td>68.92</td>
      <td>74.77</td>
      <td>0.121</td>
    </tr>
    <tr>
      <td>Nanonets-OCR-s</td>
      <td>2025.06</td>
      <td>✅</td>
      <td>3B</td>
      <td>85.59</td>
      <td>0.093</td>
      <td>85.90</td>
      <td>80.14</td>
      <td>85.57</td>
      <td>0.108</td>
    </tr>
    <tr>
      <td>Deepseek-OCR</td>
      <td>2025.10</td>
      <td>✅</td>
      <td>3B</td>
      <td>87.01</td>
      <td>0.073</td>
      <td>83.37</td>
      <td>84.97</td>
      <td>88.80</td>
      <td>0.086</td>
    </tr>
    <tr>
      <td>dots.ocr</td>
      <td>2025.07</td>
      <td>✅</td>
      <td>3B</td>
      <td>88.41</td>
      <td>0.048</td>
      <td>83.22</td>
      <td>86.78</td>
      <td>90.62</td>
      <td>0.053</td>
    </tr>
    <tr>
      <td>OCRVerse</td>
      <td>2025.10</td>
      <td>✅</td>
      <td>4B</td>
      <td>88.65</td>
      <td>0.051</td>
      <td>88.38</td>
      <td>82.67</td>
      <td>86.63</td>
      <td>0.062</td>
    </tr>
  </tbody>
</table>

### Performance Across Diverse Page Types

The following table illustrates the text recognition performance (Edit Distance) of the OCRVerse model across 9 different document types. It is intended to offer deeper insights into the model’s performance on diverse page types, thereby enabling a more nuanced understanding of its capabilities and limitations in different real-world document scenarios.

<table style="border-collapse: collapse;">
  <thead>
    <tr style="border: 1px solid black;">
      <th style="border: 1px solid black;">model</th>
      <th style="border: 1px solid black;">Book</th>
      <th style="border: 1px solid black;">PPT2PDF</th>
      <th style="border: 1px solid black;">Research Report</th>
      <th style="border: 1px solid black;">Colorful Textbook</th>
      <th style="border: 1px solid black;">Exam Paper</th>
      <th style="border: 1px solid black;">Magazine</th>
      <th style="border: 1px solid black;">Academic Literature</th>
      <th style="border: 1px solid black;">Note</th>
      <th style="border: 1px solid black;">Newspaper</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border: 1px solid black;">
      <td style="border: 1px solid black;">OCRVerse</td>
      <td style="border: 1px solid black;">0.041</td>
      <td style="border: 1px solid black;">0.026</td>
      <td style="border: 1px solid black;">0.006</td>
      <td style="border: 1px solid black;">0.092</td>
      <td style="border: 1px solid black;">0.051</td>
      <td style="border: 1px solid black;">0.03</td>
      <td style="border: 1px solid black;">0.043</td>
      <td style="border: 1px solid black;">0.069</td>
      <td style="border: 1px solid black;">0.098</td>
    </tr>
  </tbody>
</table>

### Performance Across Diverse Layouts

End-to-end reading order evaluation on OmniDocBench: results across different column layout types using Normalized Edit Distance.

<table style="border-collapse: collapse;">
  <thead>
    <tr style="border: 1px solid black;">
      <th style="border: 1px solid black;">model</th>
      <th style="border: 1px solid black;">Single Column</th>
      <th style="border: 1px solid black;">Double Column</th>
      <th style="border: 1px solid black;">Three Column</th>
      <th style="border: 1px solid black;">Other Layout</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border: 1px solid black;">
      <td style="border: 1px solid black;">OCRVerse</td>
      <td style="border: 1px solid black;">0.022</td>
      <td style="border: 1px solid black;">0.042</td>
      <td style="border: 1px solid black;">0.09</td>
      <td style="border: 1px solid black;">0.16</td>
    </tr>
  </tbody>
</table>

### Text Recognition Performance Across Attributes

The following table illustrates the text recognition performance (Edit Distance) of the OCRVerse model across diverse text attributes, including language, background, and rotation. It is intended to offer deeper insights into the model’s performance under different text properties, thereby enabling a more nuanced understanding of its capabilities and limitations in real-world document scenarios.

<table style="border-collapse: collapse;">
  <thead>
    <tr style="border-bottom: none;">
      <th rowspan="2" style="border: 1px solid black;">Model</th>
      <th colspan="3" style="border: 1px solid black;">Language</th>
      <th colspan="3" style="border: 1px solid black;">Text background</th>
      <th colspan="3" style="border: 1px solid black;">Text Rotate</th>
    </tr>
    <tr>
      <th style="border: 1px solid black;">EN</th>
      <th style="border: 1px solid black;">ZH</th>
      <th style="border: 1px solid black;">Mixed</th>
      <th style="border: 1px solid black;">White</th>
      <th style="border: 1px solid black;">Single</th>
      <th style="border: 1px solid black;">Multi</th>
      <th style="border: 1px solid black;">Normal</th>
      <th style="border: 1px solid black;">Rotate270</th>
      <th style="border: 1px solid black;">Horizontal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid black;">OCRVerse</td>
      <td style="border: 1px solid black;">0.077</td>
      <td style="border: 1px solid black;">0.084</td>
      <td style="border: 1px solid black;">0.062</td>
      <td style="border: 1px solid black;">0.081</td>
      <td style="border: 1px solid black;">0.068</td>
      <td style="border: 1px solid black;">0.08</td>
      <td style="border: 1px solid black;">0.078</td>
      <td style="border: 1px solid black;">0.968</td>
      <td style="border: 1px solid black;">0.232</td>
    </tr>
  </tbody>
</table>

# 🔍 Usage Example

## OCRVerse-text



### Inference

This below is a simple example of how to use OCRVerse-text for document parsing tasks.

Please first install [transformers](https://github.com/huggingface/transformers) using the following command:

```shell
pip install "transformers>=4.57.0"
```

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

# Load model
model_path = 'DocTron/OCRVerse-text'
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    dtype="auto", 
    device_map="cuda",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Prepare input with image and text
image_path = "./assets/ocrverse-text_test.jpg"
# We recommend using the following prompt to better performance, since it is used throughout the training process.
prompt = "Extract the main content from the document in the image, keeping the original structure. Convert all formulas to LaTeX and all tables to HTML."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ]
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=8192, do_sample=False)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.tokenizer.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])

# $$
# r = \frac{\alpha}{\beta} \sin \beta (\sigma_1 \pm \sigma_2)
# $$
```

### Fine-tuning

If you want to continue training based on our model, you can use [Llama Factory](https://github.com/hiyouga/LLaMA-Factory). For installation and usage of Llama Factory, please refer to its official documentation. A reference fine-tuning script with pre-specified parameters is provided below:

```shell
PROJECT_DIR=/path/to/llama_factory
cd ${PROJECT_DIR}

# Set parameters
GPUS_PER_NODE=8                  # Number of GPUs per node
NNODES=1                         # Total number of nodes
NODE_RANK=0                      # Rank of the current node (starts from 0)
MASTER_ADDR=localhost            # IP address of the master node
MASTER_PORT=12345                # Port for communication between nodes

MODEL_DIR=/path/to/ocrverse_text_model  # Path to the pre-trained OCRVerse model
DATA=/name/of/your/dataset               # Name/path of your custom dataset
OUTPUT_DIR=/path/to/output              # Directory to save fine-tuned results

# Llama Factory-based fine-tuning script
torchrun --nproc_per_node="${GPUS_PER_NODE}" --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    src/train.py \
    --model_name_or_path "$MODEL_DIR" \
    --stage sft \
    --do_train True \
    --finetuning_type full \
    --dataset "$DATA" \
    --template qwen3_vl_nothink \
    --cutoff_len 8192 \
    --preprocessing_num_workers 128 \
    --preprocessing_batch_size 256 \
    --dataloader_num_workers 128 \
    --output_dir "$OUTPUT_DIR" \
    --logging_steps 1 \
    --save_steps 5000 \
    --plot_loss True \
    --save_only_model False \
    --report_to none \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 True
```

# 📌 Acknowledgement
We sincerely appreciate [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for providing reference training framework.


<!-- ## 📖 Citation
If you find this project useful, please feel free to leave a star and cite our paper:
```
@misc{chen2025chartr1,
      title={Chart-R1: Chain-of-Thought Supervision and Reinforcement for Advanced Chart Reasoner}, 
      author={Lei Chen and Xuanle Zhao and Zhixiong Zeng and Jing Huang and Yufeng Zhong and Lin Ma},
      year={2025},
      eprint={2507.15509},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.15509}, 
}
``` -->
