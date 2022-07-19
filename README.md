# Quantifying Societal Bias Amplification in Image Captioning
This repository contains source code necessary to reproduce the results presented in the paper [Quantifying Societal Bias Amplification in Image Captioning](https://openaccess.thecvf.com/content/CVPR2022/html/Hirota_Quantifying_Societal_Bias_Amplification_in_Image_Captioning_CVPR_2022_paper.html) (CVPR 2022, Oral). Please check the project website [here](https://sites.google.com/view/cvpr-2022-quantify-bias/home).
## Introduction
We study societal bias amplification in image captioning. Image captioning models have been shown to perpetuate gender and racial biases, however, metrics to measure, quantify, and evaluate the societal bias in captions are not yet standardized. We provide a comprehensive study on the strengths and limitations of each metric, and propose LIC, a metric to study captioning bias amplification. We argue that, for image captioning, it is not enough to focus on the correct prediction of the protected attribute, and the whole context should be taken into account. We conduct extensive evaluation on traditional and state-of-the-art image captioning models, and surprisingly find that, by only focusing on the protected attribute prediction, bias mitigation models are unexpectedly amplifying bias.
(add a picture)

## Setup
1. Clone the repository.
2. Download the [data](https://drive.google.com/drive/folders/1PI03BqcnhdXZi2QY9PUHzWn4cxgdonT-?usp=sharing) and place in the current directory.
3. Install dependancies:
  ### For LSTM classifier
    - Python 3.7
    - numpy 
    - pytorch 1.9
    - torchtext 0.10.0 
    - spacy 
    - sklearn 
  ### For BERT classifier
    - Python 3.7
    - numpy
    - pytorch 1.4
    - transformers 4.0.1
    - spacy 2.3
    - sklearn
    
## Evaluation
(add an explanation of what is doing here)
  ### For LSTM classifier
  For training the classifier and calculating LIC on human captions in terms of gender bias.   
    - sh run_gender_lstm_ann.sh 
  For training the classifier and calculating LIC on generated captions by captioning models in terms of gender bias.
    - sh run_gender_lstm_model.sh 
  For training the classifier and calculating LIC on human captions in terms of racial bias.
    - sh run_race_lstm_ann.sh 
  For training the classifier and calculating LIC on generated captions by captioning models in terms of racial bias.
    - sh run_race_lstm_model.sh 
  ### For BERT classifier
  For training the classifier and calculating LIC on human captions in terms of gender bias. 
    - sh run_gender_bert_ann.sh 
  For training the classifier and calculating LIC on generated captions by captioning models in terms of gender bias.
    - sh run_gender_bert_model.sh 
  For training the classifier and calculating LIC on human captions in terms of racial bias.
    - sh run_race_bert_ann.sh 
  For training the classifier and calculating LIC on generated captions by captioning models in terms of racial bias.
    - sh run_race_bert_model.sh 
  ### For BERT classifier (BERT is not finetuned)
  For training the classifier and calculating LIC on human captions in terms of gender bias. 
    - sh run_gender_bert_freeze_ann.sh 
  For training the classifier and calculating LIC on generated captions by captioning models in terms of gender bias.
    - sh run_gender_bert_freeze_model.sh 
  For training the classifier and calculating LIC on human captions in terms of racial bias.
    - sh run_race_bert_freeze_ann.sh 
  For training the classifier and calculating LIC on generated captions by captioning models in terms of racial bias.
    - sh run_race_bert_freeze_model.sh 
    
## Results
<div align="center">
<img src="run_scripts/LIC_gender.png" width="600pix"/>
</div>

## Citation
    @inproceedings{hirota2022quantifying,
      title={Quantifying Societal Bias Amplification in Image Captioning},
      author={Hirota, Yusuke and Nakashima, Yuta and Garcia, Noa},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={13450--13459},
      year={2022}
     }
