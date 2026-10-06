# CVB_dataset
CVB: A Video Dataset of Cattle Visual Behaviors

Existing image/video datasets for cattle behavior recognition are mostly small, lack well-defined labels, or are collected in unrealistic controlled environments. This limits the utility of machine learning (ML) models learned from them. Therefore, we introduce a new dataset, called Cattle Visual Behaviors (CVB), that consists of 502 video clips, each fifteen seconds long, captured in natural lighting conditions, and annotated with eleven visually perceptible behaviors of grazing cattle. We use the Computer Vision Annotation Tool (CVAT) to collect our annotations. To make the procedure more efficient, we perform an initial detection and tracking of cattle in the videos using appropriate pre-trained models. The results are corrected by domain experts along with cattle behavior labeling in CVAT. The pre-hoc detection and tracking step significantly reduces the manual annotation time and effort. Moreover, we convert CVB to the atomic visual action (AVA) format and train and evaluate the popular SlowFast action recognition model on it. The associated preliminary results confirm that we can localize the cattle and recognize their frequently occurring behaviors with confidence. By creating and sharing CVB, our aim is to develop improved models capable of recognizing all important behaviors accurately and to assist other researchers and practitioners in developing and evaluating new ML models for cattle behavior classification using video data. 

The dataset is available at https://doi.org/10.25919/3g3t-p068.

The paper is available at https://arxiv.org/abs/2305.16555.

The poster is available at [here](https://drive.google.com/file/d/1CA6VxUeEhWeLB_nfo01EfzhcKmCXOHUd/view?usp=drive_link).
