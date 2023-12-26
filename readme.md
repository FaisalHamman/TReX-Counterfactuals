
# Robust Counterfactual Explanations for Neural Networks With Probabilistic Guarantees

Welcome to the  repository for the paper "Robust Counterfactual Explanations for Neural Networks With Probabilistic Guarantees." This research was presented at the International Conference on Machine Learning (ICML 2023) and is a collaborative effort between [Faisal Hamman](https://www.faisalhamman.com/), [Erfaun Noorani](https://enoorani.github.io/), [Saumitra Mishra](https://scholar.google.co.uk/citations?user=On6E6ogAAAAJ&hl=en), [Daniele Magazzeni](https://scholar.google.com/citations?user=IWXDluMAAAAJ&hl=en), and [Sanghamitra Dutta](https://sites.google.com/site/sanghamitraweb/). For further details, please refer to our [paper](https://arxiv.org/abs/2305.11997).

## Installation

Begin by setting up your environment to run the code associated with our paper:

```bash
pip install datalib-dev
pip install foolbox
pip install adversarial-robustness-toolbox
pip install carla-recourse
```

## Usage

To replicate the experiments and see our TReX method in action alongside various baselines, simply execute:

```bash
python main.py --dataset [DATASET] --tau [TAU] --sig [SIG] --max_steps [MAX_STEPS] --changed_models_size [CHANGED_MODELS_SIZE] --norm [NORM]
```

We have leveraged the CARLA - Counterfactual And Recourse Library for our baseline implementations. For more information, visit [CARLA on GitHub](https://github.com/carla-recourse/CARLA). 

## Citation

If our work contributes to your research, please consider citing it as follows:

```bibtex
@InProceedings{pmlr-v202-hamman23a,
  title = 	 {Robust Counterfactual Explanations for Neural Networks With Probabilistic Guarantees},
  author =       {Hamman, Faisal and Noorani, Erfaun and Mishra, Saumitra and Magazzeni, Daniele and Dutta, Sanghamitra},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {12351--12367},
  year = 	 {2023},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/hamman23a/hamman23a.pdf},
  url = 	 {https://proceedings.mlr.press/v202/hamman23a.html},
}
```

We also acknowledge the work of others that our code builds upon:

```bibtex
@inproceedings{
black2022consistent,
title={Consistent Counterfactuals for Deep Models},
author={Emily Black and Zifan Wang and Matt Fredrikson},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=St6eyiTEHnG}
}

@misc{pawelczyk2021carla,
      title={CARLA: A Python Library to Benchmark Algorithmic Recourse and Counterfactual Explanation Algorithms},
      author={Martin Pawelczyk and Sascha Bielawski and Johannes van den Heuvel and Tobias Richter and Gjergji Kasneci},
      year={2021},
      eprint={2108.00783},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

This project is open-sourced under the MIT License. For more details, please refer to the [LICENSE](LICENSE.md) file.

--- 
