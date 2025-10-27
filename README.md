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

## 📢 News and Updates
* ```2025.10.28``` We upload our model weights [OCRVerse-text](https://huggingface.co/DocTron/OCRVerse-text) and [OCRVerse-code](https://huggingface.co/DocTron/OCRVerse-code) to HuggingFace.
<!-- * ```2025.07.21``` 🔥🔥🔥 We release the technical report of **Chart-R1** at arXiv [link](https://arxiv.org/abs/2507.15509). -->


## 🤗 Models
|  Model   | Download Link  |
|  ----  | ----  |
|  OCRVerse-text |  [DocTron/OCRVerse-text](https://huggingface.co/DocTron/OCRVerse-text)  |
|  OCRVerse-code  |  [DocTron/OCRVerse-code](https://huggingface.co/DocTron/OCRVerse-code)   |

<!-- The ```Chart-COT``` is Qwen2.5-VL-7B-Instruct fine-tuned with supervised learning on the ChartRQA-SFT dataset. The ```Chart-R1``` is Chart-COT further optimized through reinforcement fine-tuning (RFT). -->


## 📊 Performance

### OCRVerse-text

<table>
  <thead>
    <tr>
      <th>Model Type</th>
      <th>Methods</th>
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
    <tr>
      <td rowspan="3">Pipeline Tools</td>
      <td>Marker-1.8.2</td>
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
      <td>-</td>
      <td>86.73</td>
      <td>0.073</td>
      <td>85.79</td>
      <td>81.68</td>
      <td>89.48</td>
      <td>0.073</td>
    </tr>
    <tr>
      <td rowspan="5">General VLMs</td>
      <td>GPT-4o</td>
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
      <td>-</td>
      <td>88.03</td>
      <td>0.075</td>
      <td>85.82</td>
      <td>85.71</td>
      <td>90.29</td>
      <td>0.097</td>
    </tr>
    <tr>
      <td rowspan="7">Layout-aware VLMs</td>
      <td>Dolphin</td>
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
      <td>0.9B</td>
      <td>92.56</td>
      <td>0.035</td>
      <td>91.43</td>
      <td>89.76</td>
      <td>93.52</td>
      <td>0.043</td>
    </tr>
    <tr>
      <td rowspan="7">End-to-End VLMs</td>
      <td>OCRFlux-3B</td>
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
      <td>3B</td>
      <td>85.59</td>
      <td>0.093</td>
      <td>85.90</td>
      <td>80.14</td>
      <td>85.57</td>
      <td>0.108</td>
    </tr>
    <tr>
      <td>dots.ocr</td>
      <td>3B</td>
      <td>88.41</td>
      <td>0.048</td>
      <td>83.22</td>
      <td>86.78</td>
      <td>90.62</td>
      <td>0.053</td>
    </tr>
      <td>OCRVerse-text</td>
      <td>4B</td>
      <td>87.91</td>
      <td>0.052</td>
      <td>87.11</td>
      <td>81.81</td>
      <td>86.14</td>
      <td>0.065</td>
    </tr>
  </tbody>
</table>

### OCRVerse-code


<!-- ## 🔍 Usage Example
Below is a simple example of how to use Chart-R1 for multimodal reasoning tasks:
```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load model
model_path = 'DocTron/Chart-R1'

# Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2"
)

# Use the following system_prompt and pixel range by default
system_prompt = "Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since 1+1=2, so the answer is 2. </think><answer> 2 </answer>, which means assistant's output should start with <think> and end with </answer>."

processor = AutoProcessor.from_pretrained(model_path, min_pixels=1280*28*28, max_pixels=16384*28*28)

# Set generation parameters by default
generate_kwargs = dict(
    max_new_tokens=2048,
    top_p=0.001,
    top_k=1,
    temperature=0.01,
    repetition_penalty=1.0
)

# Prepare input with image and text
messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "assets/example_case.jpg",
            },
            {"type": "text", "text": "What is the difference in percentage of U.S. people who thinks scientists should take active part in policy debates and those thinks they should focus on establishing sound scientific facts?"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, **generate_kwargs)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])

# <think>Step 1: Identify the percentage of U.S. people who think scientists should 'Take an active role in policy debates'. This is 60%. Step 2: Identify the percentage of U.S. people who think scientists should 'Focus on establishing sound scientific facts'. This is 39%. Step 3: Calculate the difference between these two percentages: 60% - 39% = 21%.</think><answer>21</answer>
``` -->


## 📌 Acknowledgement
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
