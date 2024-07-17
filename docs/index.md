# <center> Training a shadow removal network using only 3D primitive occluders
### <center>In The Visual Computer - Springer</center>
### <center>Neil Patrick Del Gallego, Joel Ilao, Macario Cordel II, Conrado Ruiz Jr.</center>
### <center>De La Salle University, Grup de Recerca en Tecnologies Media, La Salle - Universitat Ramon Llull</center>

<center><img src="web_img/logo_1.png" width="512"></center>

<br>

### <strong><a href = "https://link.springer.com/article/10.1007/s00371-024-03536-7"> Paper </a> | <a href = "https://github.com/NeilDG/SynthShadowRemoval">Source code</a> </strong>
<img src="web_img/results_1.png" width=1949px/>
<img src="web_img/results_22.png"/>
### Abstract

<p align="justify"> Removing shadows in images is often a necessary pre-processing task for improving the performance of computer vision applications. Deep learning shadow removal approaches require a large-scale dataset that is challenging to gather. To address the issue of limited shadow data, we present a new and cost-effective method of synthetically generating shadows using 3D virtual primitives as occluders. We simulate the shadow generation process in a virtual environment where foreground objects are composed of mapped textures from the Places-365 dataset. We argue that complex shadow regions can be approximated by mixing primitives, analogous to how 3D models in computer graphics can be represented as triangle meshes. We use the proposed synthetic shadow removal dataset, DLSUSynthPlaces-100K, to train a feature-attention-based shadow removal network without explicit domain adaptation or style transfer strategy. The results of this study show that the trained network achieves competitive results with state-of-the-art shadow removal networks that were trained purely on typical SR datasets such as ISTD or SRD. Using a synthetic shadow dataset of only triangular prisms and spheres as occluders produces the best results. Therefore, the synthetic shadow removal dataset can be a viable alternative for future deep-learning shadow removal methods. </p>

### Shadow-Free Image Results
We provide our shadow-free image results on the ISTD and SRD datasets.

<br>
<a href = "https://drive.google.com/file/d/1w4ENpcU1y1zEOY24yGAlIURLrzB7ppKw/view">ISTD and SRD image results</a>
  
### DLSU-SynthPlaces100K (SYNthetic Shadows on Places-365)
Training images used in our paper: <a href = "">DLSU-SynthPlaces100K Dataset (COMING SOON) </a> <br>
All images are numbered and paired. You can find each of these images, in their corresponding folders. Example: ```synth_0.png``` <br>

  
### Pre-Trained Models
Shadow-matte pre-trained network - labelled Gm, in our paper: <a href = "https://drive.google.com/file/d/1wtyN271B7jKaLfQ1En0cVOr0VAJmFxl5/view?usp=sharing">rgb2sm.pth </a>
<br>
Shadow removal pre-trained network - labelled Gz, in our paper: <a href = "https://drive.google.com/file/d/1wmwcRv5olkBum0FUwMc4Z_Jy9oUp6rQB/view?usp=sharing">rgb2ns.pth </a>

<br>
Assuming you have the source project, place all models in <b>"./checkpoint" </b> directory.

### Training
Our training pipeline is divided into two: training the shadow-matte network (```shadow_train_main.py```), then training the shadow removal network (```shadow_test_main.py```). 
A sample training sequence can be found in ```rtx_3090_main.py``` <br>

We do not train the networks in the cloud, and have used several RTX-based PCs. Thus, several configurations (e.g. directories, batch sizes) are hard-coded as such configurations are known and fixed beforehand.
We believe our code is readable enough to understand our training pipeline.

### Inference
Assuming you already have the pre-trained models, you can perform inference by running 
```
python  shadow_test_main.py  --...<check parameters supported in our source code>"
```

An example inference script is located in ```test_main.py```.

### Citation
```
@article{gallego2024training,
  title={Training a shadow removal network using only 3D primitive occluders},
  author={Gallego, Neil Patrick Del and Ilao, Joel and Cordel, Macario II and Ruiz, Conrado},
  journal={The Visual Computer},
  pages={1--38},
  year={2024},
  publisher={Springer}
}
```

### Acknowledgements
We would like to acknowledge De La Salle University (DLSU), Department of Science and Technology (DOST), and the Google Cloud Research program, for funding this research.
