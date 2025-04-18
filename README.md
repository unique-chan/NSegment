<h1 align="center">
  N<sup>Segment</sup>: Noisy Segment Improves Remote Sensing Image Segmentation
</h1>

<p align="center">
  <a href="#"><img alt="Python3.7+" src="https://img.shields.io/badge/Python-3.7+-blue?logo=python&logoColor=white"></a>
  <a href="#"><img alt="PyTorch1.6+" src="https://img.shields.io/badge/PyTorch-1.6+-orange?logo=pytorch&logoColor=white"></a>
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
- Official implementation of our proposed approach

### Announcement:
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
    def __init__(self, alpha_sigma_list, prob=0.5):
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
        alpha, sigma = self.alpha_sigma_list[random_state.randint(0, len(alpha_sigma_list))]
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

### Preliminaries:
- Install all necessary packages listed in the `requirements.txt`. 
- Simply add our `NoisySegment` to *train_pipeline* in your model configuration file. Below is an example:
~~~python3
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='NoisySegment'), # Our transformation (⭐) should be placed directly after 'LoadAnnotations'
    ...
]
~~~

### Training via our strategy:
- ...

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

### Contribution:
If you find any bugs or have opinions for further improvements, please feel free to create a pull request or contact me (yechankim@gm.gist.ac.kr). All contributions are welcome.
