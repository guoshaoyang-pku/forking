---
dataset_info:
- config_name: default
  features:
  - name: system
    dtype: string
  - name: conversations
    list:
    - name: from
      dtype: string
    - name: value
      dtype: string
  splits:
  - name: train
    num_bytes: 2635015668
    num_examples: 113957
  download_size: 1078777193
  dataset_size: 2635015668
- config_name: metadata
  features:
  - name: problem
    dtype: string
  - name: deepseek_reasoning
    dtype: string
  - name: deepseek_solution
    dtype: string
  - name: ground_truth_solution
    dtype: string
  - name: domain
    dtype: string
  - name: source
    dtype: string
  - name: test_cases
    dtype: string
  - name: starter_code
    dtype: string
  splits:
  - name: train
    num_bytes: 5525214077.699433
    num_examples: 113957
  download_size: 2469729724
  dataset_size: 5525214077.699433
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
- config_name: metadata
  data_files:
  - split: train
    path: metadata/train-*
tags:
- curator
- synthetic
license: apache-2.0
---

<p align="center">
    <img src="open_thoughts.png" width="50%">
</p>

> [!NOTE]
> We have released a paper for OpenThoughts! See our paper [here](https://arxiv.org/abs/2506.04178).

<a href="https://github.com/bespokelabsai/curator/">
 <img src="https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k/resolve/main/made_with_curator.png" alt="Made with Curator" width=200px>
</a>

# Open-Thoughts-114k

## Dataset Description

- **Homepage:** https://www.open-thoughts.ai/
- **Repository:** https://github.com/open-thoughts/open-thoughts
- **Point of Contact:** [Open Thoughts Team](contact@open-thoughts.ai)

Open synthetic reasoning dataset with 114k high-quality examples covering math, science, code, and puzzles!

Inspect the content with rich formatting with [Curator Viewer](https://curator.bespokelabs.ai/datasets/1389c194254c4ead96daaf145505c3d1).

### Available Subsets

**default** subset containing ready-to-train data used to finetune the [OpenThinker-7B](https://huggingface.co/open-thoughts/OpenThinker-7B) and [OpenThinker-32B](https://huggingface.co/open-thoughts/OpenThinker-32B) models:
```
ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
```
**metadata** subset containing extra columns used in dataset construction:
- `problem`
- `ground_truth_solution`
- `deepseek_reasoning`
- `deepseek_solution`
- `domain`
- `source`
- `test_cases` (code only)
- `starter_code`(code only)

```
ds = load_dataset("open-thoughts/OpenThoughts-114k", "metadata", split="train")
```

# OpenThinker Models

The numbers reported in the tables below are evaluated with our open-source tool [Evalchemy](https://github.com/mlfoundations/Evalchemy).

|                             | AIME24   | MATH500 | GPQA-Diamond | LCBv2 Easy  | LCBv2 Medium  | LCBv2 Hard  | LCBv2 All  |
| --------------------------- | -------- | ------- | ------------ | ----------- | ------------- | ----------- | ---------- |
| [OpenThinker-32B](https://huggingface.co/open-thoughts/OpenThinker-32B)             | 66       | 90.6    | 61.6         | 95.1        | 70.9          | 26.8        | 68.9       |
| [OpenThinker-7B](https://huggingface.co/open-thoughts/OpenThinker-7B)              | 31.3     | 83.0    | 42.4         | 75.3        | 28.6          | 6.5         | 39.9       |
| Bespoke-Stratos-7B          | 22.7     | 79.6    | 38.9         | 71.4        | 25.2          | 0.8         | 35.8       |
| DeepSeek-R1-Distill-Qwen-7B | 60       | 88.2    | 46.9         | 79.7        | 45.1          | 14.6        | 50.1       |
| gpt-4o-0513                 | 8.7      | 75.8    | 46.5         | 87.4        | 42.7          | 8.9         | 50.5       |
| o1-mini                     | 64       | 85.6    | 60           | 92.8        | 74.7          | 39.8        | 72.8       |

We are fully open-source. Our [model weights](https://huggingface.co/open-thoughts), [datasets](https://huggingface.co/open-thoughts), [data generation code](https://github.com/open-thoughts/open-thoughts), [evaluation code](https://github.com/mlfoundations/Evalchemy), and [training code](https://github.com/hiyouga/LLaMA-Factory) are all publicly available. 

|  | Open Weights | Open Data | Open Code | 
|--|--------------|-----------| --------- |
|OpenThinker-32B|✅|[✅](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)|[✅](https://github.com/open-thoughts/open-thoughts) |
|OpenThinker-7B|✅|[✅](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)|[✅](https://github.com/open-thoughts/open-thoughts) |
|Bespoke-Stratos-7B|✅|[✅](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k)|[✅](https://github.com/bespokelabsai/curator/tree/main/examples/bespoke-stratos-data-generation)|
|DeepSeek-R1-Distill models|✅|❌|❌|
|OpenAI/Gemini|❌|❌|❌|❌|

We are actively working towards improving the dataset, so please stay tuned!

# Data Curation Recipe

Code
- [BAAI/TACO](https://huggingface.co/datasets/BAAI/TACO)
- [codeparrot/apps](https://huggingface.co/datasets/codeparrot/apps)
- [deepmind/code_contests](https://huggingface.co/datasets/deepmind/code_contests)
- [MatrixStudio/Codeforces-Python-Submissions](https://huggingface.co/datasets/MatrixStudio/Codeforces-Python-Submissions)

Math
- [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)

Science
- [camel-ai/chemistry](https://huggingface.co/datasets/camel-ai/chemistry)
- [camel-ai/biology](https://huggingface.co/datasets/camel-ai/biology)
- [camel-ai/physics](https://huggingface.co/datasets/camel-ai/physics)

Puzzle
- [INK-USC/riddle_sense](https://huggingface.co/datasets/INK-USC/riddle_sense)

Using a curated mix of the datasets above, we generate reasoning traces from DeepSeek-R1 and verify correctness to construct the final dataset.

![diagram](diagram.png)

The full code for the data generation pipeline is publicly available [in our github repo](https://github.com/open-thoughts/open-thoughts).


# Links
- 📝 [OpenThoughts Paper](https://arxiv.org/abs/2506.04178)
- 📊 [OpenThinker-32B Blog Post](https://www.open-thoughts.ai/blog/scale)
- 📊 [Measuing Reasoning with Evalchemy Blog Post](https://www.open-thoughts.ai/blog/measure)
- 📊 [Open Thoughts Launch Blog Post](https://www.open-thoughts.ai/blog/launch)
- 💻 [Open Thoughts GitHub Repository](https://github.com/open-thoughts/open-thoughts)
- 🧠 [OpenThoughts-114k dataset](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) - this dataset.
- 🤖 [OpenThinker-32B model](https://huggingface.co/open-thoughts/OpenThinker-32B)
- 🤖 [OpenThinker-7B model](https://huggingface.co/open-thoughts/OpenThinker-7B)
- 📊 [Bespoke-Stratos Blog Post](https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation)
- 🧠 [Bespoke-Stratos-17k dataset](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k)
- 🤖 [Bespoke-Stratos-32B model](https://huggingface.co/bespokelabs/Bespoke-Stratos-32B)
- 🤖 [Bespoke-Stratos-7B model](https://huggingface.co/bespokelabs/Bespoke-Stratos-7B)
- 💻 [Curator Viewer](https://curator.bespokelabs.ai/datasets/1389c194254c4ead96daaf145505c3d1)

## Visualization
Inspect the content with rich formatting with [Curator Viewer](https://curator.bespokelabs.ai/datasets/1389c194254c4ead96daaf145505c3d1)

All 114k examples, clustered by semantic similarity, can be explored in [Nomic Atlas](https://atlas.nomic.ai/data/nomic/openthoughts-114k/map). 

<a href="https://atlas.nomic.ai/data/nomic/openthoughts-114k/map">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/630bfb6b86b8b9904c35f4d1/d7TjezV6R3OnIDlEVL1Rl.png" alt="Nomic Atlas Open-Thoughts-114k Map" width="35%"/>
</a>

# Citation
```
@misc{guha2025openthoughtsdatarecipesreasoning,
  title={OpenThoughts: Data Recipes for Reasoning Models}, 
  author={Etash Guha and Ryan Marten and Sedrick Keh and Negin Raoof and Georgios Smyrnis and Hritik Bansal and Marianna Nezhurina and Jean Mercat and Trung Vu and Zayne Sprague and Ashima Suvarna and Benjamin Feuer and Liangyu Chen and Zaid Khan and Eric Frankel and Sachin Grover and Caroline Choi and Niklas Muennighoff and Shiye Su and Wanjia Zhao and John Yang and Shreyas Pimpalgaonkar and Kartik Sharma and Charlie Cheng-Jie Ji and Yichuan Deng and Sarah Pratt and Vivek Ramanujan and Jon Saad-Falcon and Jeffrey Li and Achal Dave and Alon Albalak and Kushal Arora and Blake Wulfe and Chinmay Hegde and Greg Durrett and Sewoong Oh and Mohit Bansal and Saadia Gabriel and Aditya Grover and Kai-Wei Chang and Vaishaal Shankar and Aaron Gokaslan and Mike A. Merrill and Tatsunori Hashimoto and Yejin Choi and Jenia Jitsev and Reinhard Heckel and Maheswaran Sathiamoorthy and Alexandros G. Dimakis and Ludwig Schmidt},
  year={2025},
  eprint={2506.04178},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2506.04178}, 
}
```
