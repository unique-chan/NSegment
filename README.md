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
    <img alt="Welcome" src="sample.png" />
</p>

### Preliminaries:
- Install all necessary packages listed in the `requirements.txt`. 
- Simply add our `NSegment` to *train_pipeline* in your model configuration file. Below is an example:
~~~python3
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='NSegment'), # Our transformation (⭐) should be placed directly after 'LoadAnnotations'
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
