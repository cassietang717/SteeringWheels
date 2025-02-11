# DATA 37712 Final Project: Steering Wheel – Text-Image Fusion for Multi-Modal LLM Steering

## Group Member
- **Dongwei Lyu** (`dwlyu`) - [dwlyu@uchicago.edu](mailto:dwlyu@uchicago.edu)
- **Yushan Tang** (`cassietang`) - [cassietang@uchicago.edu](mailto:cassietang@uchicago.edu)
- **Weiyi Tian** (`weiyitian`) - [weiyitian@uchicago.edu](mailto:weiyitian@uchicago.edu)

## MMHal with Llava-7b

* [`llava_example.py`](./MMHal/llava_example.py): example of hallucination with llava-7b on a single image data
* [`MMHal.py`](./MMHal/MMHal.py): use llava-7b to process data from MMHal-bench and generate outputs and save it to [`MMHal_output.json`](./MMHal/output/MMHal_output.json)
* [`MMHal_st_eval.py`](./MMHal/MMHal_st_eval.py): use sentence transformer to evaluate the similarity between the ground truth answer and the model answer and save it to [`MMHal_st.json`](./MMHal/output/MMHal_st.json)

## Dependencies
The project is dependent on packages specified in [`requirements.txt`](./requirements.txt).

## License
Distributed under the [MIT License](/LICENSE).

## Acknowledgments
* [MMHal-Bench](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench)
* [llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b)