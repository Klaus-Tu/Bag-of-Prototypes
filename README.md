# Bag-of-Prototypes

This codebase provides an official implementation for the paper: [A Closer Look at the Robustness of Contrastive
Language-Image Pre-Training (CLIP)](https://openreview.net/forum?id=wMNpMe0vp3) at NeurIPS 2023.

### Abstract
Contrastive Language-Image Pre-training (CLIP) models have demonstrated remarkable generalization capabilities across multiple challenging distribution shifts. However, there is still much to be explored in terms of their robustness to the variations of specific visual factors. In real-world applications, reliable and safe systems must consider other safety objectives beyond classification accuracy, such as predictive uncertainty. Yet, the effectiveness of CLIP models on such safetyrelated features is less-explored. Driven by the above, this work comprehensively investigates the safety objectives of CLIP models, specifically focusing on three key properties: resilience to visual factor variations, calibrated uncertainty estimations, and the ability to detect anomalous inputs. To this end, we study 83 CLIP models and 127 ImageNet classifiers. They are diverse in architecture, (pre)training distribution and training strategies. We consider 10 visual factors (e.g., shape and pattern), 5 types of out-of-distribution data, and 8 natural and challenging test conditions with different shift types, such as texture, style, and perturbation shifts. Our study has unveiled several previously unknown insights into CLIP models. For instance, they are not consistently more calibrated than other ImageNet models, which contradicts existing findings. Additionally, our analysis underscores the significance of training source design by showcasing its profound influence on the three key properties. We believe our comprehensive study can shed light on and help guide the development of more robust and reliable CLIP models.

## PyTorch Implementation

This repository contains:

- the Python implementation of evluation codes.

Please follow the instruction below to install it and run the experiment demo.

## Citation
 ```bibtex
@inproceedings{tu2023closer,
  title={A Closer Look at the Robustness of Contrastive
Language-Image Pre-Training (CLIP)},
  author={Tu, Weijie and Deng, Weijian and Gedeon, Tom},
  booktitle={Advances on Neural Information Processing Systems},
  year={2023}
}
```


## License
MIT
