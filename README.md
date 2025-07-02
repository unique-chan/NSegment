<h1 align="center">
  N<sup>Segment</sup>: Label-specific Deformations for Remote Sensing Image Segmentation
</h1>

<p align="center">
  <a href="#"><img alt="Python3.7+" src="https://img.shields.io/badge/Python-3.7+-blue?logo=python&logoColor=white"></a>
  <a href="#"><img alt="PyTorch1.6+(≥2.0 recommended)" src="https://img.shields.io/badge/PyTorch-1.6+ (≥2.0 recommended)-orange?logo=pytorch&logoColor=white"></a>
  <a href="#"><img alt="MMSegmentation1.2.2" src="https://img.shields.io/badge/MMSegmentation-1.2.2-red?logo=mmlab&logoColor=white"></a>
  <a href="#"><img alt="MIT" src="https://img.shields.io/badge/License-MIT-green?logo=MIT"></a>
</p>

<p align="center">
  <b>Yechan Kim*</b>, 
  <b>DongHo Yoon*</b>,
  <b>SooYeon Kim</b>, and 
  <b>Moongu Jeon</b>
</p>

### This repo includes:
- Official implementation of our proposed approach (**_NSegment_**)

### Announcement:
- Apr. 2025: Our preprint is now available at https://arxiv.org/abs/2504.19634.
- Apr. 2025: We have released the official code of our proposed approach!

### Overview:
- With our strategy, you can boost remote sensing image segmentation.
<p align="center">
    <img alt="Welcome" src="overview.png" />
</p>

### Our algorithm in Python3 (with OpenCV, Numpy, and MMSegmentation)
~~~python3
import cv2
import numpy as np

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class NoisySegment:
    def __init__(self, alpha_sigma_list=None, prob=0.5):
        if not alpha_sigma_list:
            alpha_sigma_list = [(alpha, sigma) for alpha in [1, 15, 30, 50, 100] 
                                 for sigma in [3, 5, 10]]
        self.alpha_sigma_list = alpha_sigma_list
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        segment = results['gt_seg_map']
        noisy_segment = self.transform(segment)
        results['gt_seg_map'] = noisy_segment
        return results

    def transform(self, segment, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(None)
        alpha, sigma = self.alpha_sigma_list[random_state.randint(0, len(self.alpha_sigma_list))]
        shape = segment.shape[:2]
        dx = alpha * (2 * random_state.rand(*shape) - 1)
        dy = alpha * (2 * random_state.rand(*shape) - 1)
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
        map_y = np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)
        noisy_segment = cv2.remap(segment, map_x, map_y, interpolation=cv2.INTER_NEAREST)
        return noisy_segment
~~~

### How to use:
* If you use our git repository, our augmentation method is already included and registered.
* Simply add our `NoisySegment` to *train_pipeline* in your model configuration file. Below is an example:
  ~~~python3
  train_pipeline = [
      dict(type='LoadImageFromFile'),
      dict(type='LoadAnnotations'),
      dict(type='NoisySegment'), # ⭐
      ...
  ]
  ~~~
* For better training, you might adjust various $(\alpha, \sigma)$ parameter pairs via `alpha_sigma_list` as follows:
  ~~~python3
  dict(type='NoisySegment', alpha_sigma_list=[(1, 3), (1, 5) ...])
  ~~~
* If you want our method be applied with a prob of 80%? (Note: default prob = 0.5)
  ~~~python3
  dict(type='NoisySegment', alpha_sigma_list=[(1, 3), (1, 5) ...], prob=0.8)
  ~~~

### Preliminaries:
* **Step 1**. Create a conda environment with Python 3.8 and activate it.
    ~~~shell
    conda create -n nsegment python=3.8 -y
    conda activate nsegment
    ~~~

* **Step 2.** Install PyTorch with TorchVision following [official instructions](https://pytorch.org/get-started/locally/). The below is an example. 
    ~~~shell
    conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=12.1 -c pytorch -c nvidia
    ~~~

* **Step 3.** Install `MMSegmentation (v1.2.2)` ([v1.2.2](https://mmsegmentation.readthedocs.io/en/latest/overview.html) is the latest version of 2024).
    ~~~shell
    # ⚠️ No need to clone MMSeg (e.g. "git clone https://github.com/open-mmlab/mmsegmentation; rm -rf mmsegmentation/.git"). Already cloned! 
    pip install -U openmim==0.3.9
    mim install mmengine==0.10.7
    mim install mmcv==2.1.0
    pip install -v -e mmsegmentation/
    pip install ftfy==6.2.3
    pip install regex==2024.11.6
    ~~~

### Citation:
If you use this code for your research, please cite the following paper:
- For Latex:
  ~~~ME
  @article{kim2025nsegment,
    title={NSegment: Noisy Segment Improves Remote Sensing Image Segmentation},
    author={Kim, Yechan and Yoon, DongHo and Kim, SooYeon and Jeon, Moongu},
    journal={Arxiv Preprint},
  }
  ~~~

- For Word (MLA Style):
  ~~~ME
  Yechan Kim, DongHo Yoon, SooYeon Kim, and Moongu Jeon. "NSegment: Noisy Segment Improves Remote Sensing Image Segmentation." Arxiv Preprint (2025).
  ~~~

We also highly recommend referring to our prior study as follows:
- GitHub: https://github.com/unique-chan/NBBOX
- For Latex:
  ~~~ME
  @article{kim2025nbbox,
    title={NBBOX: Noisy Bounding Box Improves Remote Sensing Object Detection},
    author={Kim, Yechan and Kim, SooYeon and Jeon, Moongu},
    journal={IEEE Geoscience and Remote Sensing Letters},
    volume={22},
    year={2025},
    publisher={IEEE}
  }
  ~~~

- For Word (MLA Style):
  ~~~ME
  Yechan Kim, SooYeon Kim, and Moongu Jeon. "NBBOX: Noisy Bounding Box Improves Remote Sensing Object Detection." IEEE Geoscience and Remote Sensing Letters 22 (2025).
  ~~~

### Contribution:
If you find any bugs or have opinions for further improvements, please feel free to create a pull request or contact me (yechankim@gm.gist.ac.kr). All contributions are welcome.
