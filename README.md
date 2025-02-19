# DATA 37712 Final Project: Steering Wheel – Text-Image Fusion for Multi-Modal LLM Steering

## Group Member
- **Dongwei Lyu** (`dwlyu`) - [dwlyu@uchicago.edu](mailto:dwlyu@uchicago.edu)
- **Yushan Tang** (`cassietang`) - [cassietang@uchicago.edu](mailto:cassietang@uchicago.edu)
- **Weiyi Tian** (`weiyitian`) - [weiyitian@uchicago.edu](mailto:weiyitian@uchicago.edu)

## MMHal with Llava-7b

* [`llava_example.py`](./MMHal/llava_example.py): example of hallucination with llava-7b on a single image data
* [`MMHal.py`](./MMHal/MMHal.py): use llava-7b to process data from MMHal-bench and generate outputs and save it to [`MMHal_output.json`](./MMHal/output/MMHal_output.json)
* [`MMHal_st_eval.py`](./MMHal/MMHal_st_eval.py): use sentence transformer to evaluate the similarity between the ground truth answer and the model answer and save it to [`MMHal_st.json`](./MMHal/output/MMHal_st.json)
* [`MMHal_llama_eval.py`](./MMHal/MMHal_llama_eval.py): use llama-3.1-8b to evaluate the hallucination between the ground truth answer and the model answer and save it to [`MMHal_llava.json`](./MMHal/output/MMHal_llava.json)

## Collect Activation Package
### text dataset
- [Truthful_QA](https://huggingface.co/datasets/truthfulqa/truthful_qa)
- [HaluEval](https://github.com/RUCAIBox/HaluEval/tree/main/data)
    - Summary
    - Question & Answer
### Baseline model
```
cd get_activations

python get_activations.py --dataset_name halu_qa --save 0 --dataset_length 300
```


## Dependencies
The project is dependent on packages specified in [`requirements.txt`](./requirements.txt).

## License
Distributed under the [MIT License](/LICENSE).

## Acknowledgments
* [sentence transformer](https://huggingface.co/sentence-transformers)
* [llama-3.1-8b-instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
* [llava-v1.5-7b](https://github.com/haotian-liu/LLaVA/tree/main)
* [MMHal-Bench](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench)
* [Truthful_QA](https://huggingface.co/datasets/truthfulqa/truthful_qa)
* [HaluEval](https://github.com/RUCAIBox/HaluEval/tree/main/data)
* [HaloQuest](https://github.com/google/haloquest)