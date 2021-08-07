# Awesome-Transformer-CV

If you have any problems, suggestions or improvements, please submit the issue or PR.

## Contents
* [Attention](#attention)
* [OverallSurvey](#OverallSurvey)
* [NLP](#nlp)
    * [Language](#language)
    * [Speech](#Speech)
* [CV](#cv)
    * [Backbone_Classification](#Backbone_Classification)
    * [Self-Supervised](#Self-Supervised)
    * [Interpretability and Robustness](#Interpretability-and-Robustness)
    * [Detection](#Detection)
    * [HOI](#HOI)
    * [Tracking](#Tracking)
    * [Segmentation](#Segmentation)
    * [Reid](#Reid)
    * [Localization](#Localization)
    * [Generation](#Generation)
    * [Inpainting](#Inpainting)
    * [Image enhancement](#Image-enhancement)
    * [Pose Estimation](#Pose-Estimation)
    * [Face](#Face)
    * [Video Understanding](#Video-Understanding)
    * [Depth Estimation](#Depth-Estimation)
    * [Prediction](#Prediction)
    * [NAS](#NAS)
    * [PointCloud](#PointCloud)
    * [Fashion](#Fashion)
    * [Medical](#Medical)
* [Cross-Modal](#Cross-Modal)
* [Reference](#Reference)
* [Acknowledgement](#Acknowledgement)


## Attention
- Recurrent Models of Visual Attention [2014 deepmind NIPS]
- Neural Machine Translation by Jointly Learning to Align and Translate [ICLR 2015]

## OverallSurvey
- Efficient Transformers: A Survey [[paper](https://arxiv.org/abs/2009.06732)]
- A Survey on Visual Transformer [[paper](https://arxiv.org/abs/2012.12556)]
- Transformers in Vision: A Survey [[paper](https://arxiv.org/abs/2101.01169)]
 
## NLP

<a name="language"></a>
### Language
- Sequence to Sequence Learning with Neural Networks [NIPS 2014] [[paper](https://arxiv.org/abs/1409.3215)] [[code](https://github.com/bentrevett/pytorch-seq2seq)]
- End-To-End Memory Networks [NIPS 2015] [[paper](https://arxiv.org/abs/1503.08895)] [[code](https://github.com/nmhkahn/MemN2N-pytorch)]
- Attention is all you need [NIPS 2017] [[paper](https://arxiv.org/abs/1706.03762)] [[code]()]
- **B**idirectional **E**ncoder **R**epresentations from **T**ransformers: BERT [[paper]()] [[code](https://huggingface.co/transformers/)] [[pretrained-models](https://huggingface.co/transformers/pretrained_models.html)]
- Reformer: The Efficient Transformer [ICLR2020] [[paper](https://arxiv.org/abs/2001.04451)] [[code](https://github.com/lucidrains/reformer-pytorch)]
- Linformer: Self-Attention with Linear Complexity [AAAI2020] [[paper](https://arxiv.org/abs/2006.04768)] [[code](https://github.com/lucidrains/linformer)]
- GPT-3: Language Models are Few-Shot Learners [NIPS 2020] [[paper](https://arxiv.org/abs/2005.14165)] [[code](https://github.com/openai/gpt-3)]

<a name="Speech"></a>
### Speech
- Dual-Path Transformer Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation [INTERSPEECH 2020] [[paper](https://arxiv.org/abs/2007.13975)] [[code](https://github.com/ujscjj/DPTNet)]

## CV
<a name="Backbone_Classification"></a>
### Backbone_Classification
#### Papers and Codes
- CoaT: Co-Scale Conv-Attentional Image Transformers [arxiv 2021] [[paper](http://arxiv.org/abs/2104.06399)] [[code](https://github.com/mlpc-ucsd/CoaT)]
- SiT: Self-supervised vIsion Transformer [arxiv 2021] [[paper](https://arxiv.org/abs/2104.03602)] [[code](https://github.com/Sara-Ahmed/SiT)]
- VIT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale [VIT] [ICLR 2021] [[paper](https://arxiv.org/abs/2010.11929)] [[code](https://github.com/lucidrains/vit-pytorch)]
    - Trained with extra private data: do not generalized well when trained on insufficient amounts of data
- DeiT: Data-efficient Image Transformers [arxiv2021] [[paper](https://arxiv.org/abs/2012.12877)] [[code](https://github.com/facebookresearch/deit)]
    - Token-based strategy and build upon VIT and convolutional models
- Transformer in Transformer [arxiv 2021] [[paper](https://arxiv.org/abs/2103.00112)] [[code1](https://github.com/lucidrains/transformer-in-transformer)] [[code-official](https://github.com/huawei-noah/noah-research/tree/master/TNT)]
- OmniNet: Omnidirectional Representations from Transformers [arxiv2021] [[paper](https://arxiv.org/abs/2103.01075)]
- Gaussian Context Transformer [CVPR 2021] [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ruan_Gaussian_Context_Transformer_CVPR_2021_paper.pdf)]
- General Multi-Label Image Classification With Transformers [CVPR 2021] [[paper](https://arxiv.org/abs/2011.14027)] [[code](https://github.com/QData/C-Tran)]
- Scaling Local Self-Attention for Parameter Efficient Visual Backbones [CVPR 2021] [[paper](https://arxiv.org/abs/2103.12731)]
- T2T-ViT: Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet [ICCV 2021] [[paper](https://arxiv.org/abs/2101.11986)] [[code](https://github.com/yitu-opensource/T2T-ViT)]
- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows [ICCV 2021] [[paper](https://arxiv.org/abs/2103.14030)] [[code](https://github.com/microsoft/Swin-Transformer)]
- Bias Loss for Mobile Neural Networks [ICCV 2021] [[paper](https://arxiv.org/abs/2107.11170)] [[code()]]
- Vision Transformer with Progressive Sampling [ICCV 2021] [[paper](https://arxiv.org/abs/2108.01684)] [[code(https://github.com/yuexy/PS-ViT)]]
- Rethinking Spatial Dimensions of Vision Transformers [ICCV 2021] [[paper](https://arxiv.org/abs/2103.16302)] [[code](https://github.com/naver-ai/pit)]
- Rethinking and Improving Relative Position Encoding for Vision Transformer [ICCV 2021] [[paper](https://arxiv.org/abs/2107.14222)] [[code](https://github.com/microsoft/AutoML/tree/main/iRPE)]

#### Interesting Repos
- [Convolutional Cifar10](https://github.com/kuangliu/pytorch-cifar/blob/master/main.py)
- [vision-transformers-cifar10](https://github.com/kentaroy47/vision-transformers-cifar10)
    - Found that performance was worse than simple resnet18
    - The influence of hyper-parameters: dim of vit, etc.
- [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
    - Using pretrained weights can get better results

<a name="Self-Supervised"></a>
### Self-Supervised
- Emerging Properties in Self-Supervised Vision Transformers [ICCV 2021] [[paper](https://arxiv.org/abs/2104.14294)] [[code](https://github.com/facebookresearch/dino)]
- An Empirical Study of Training Self-Supervised Vision Transformers [ICCV 2021] [[paper](https://arxiv.org/abs/2104.02057)] [[code](https://github.com/searobbersduck/MoCo_v3_pytorch)]

<a name="Interpretability-and-Robustness"></a>
### Interpretability and Robustness
- Transformer Interpretability Beyond Attention Visualization [CVPR 2021] [[paper](https://arxiv.org/abs/2012.09838)] [[code](https://github.com/hila-chefer/Transformer-Explainability)]
- On the Adversarial Robustness of Visual Transformers [arxiv 2021] [[paper](https://arxiv.org/abs/2103.15670)] 
- Robustness Verification for Transformers [ICLR 2020] [[paper](https://arxiv.org/abs/2002.06622)] [[code](https://github.com/shizhouxing/Robustness-Verification-for-Transformers)]
- Pretrained Transformers Improve Out-of-Distribution Robustness [ACL 2020] [[paper](https://arxiv.org/abs/2004.06100)] [[code](https://github.com/camelop/NLP-Robustness)]

<a name="Detection"></a>
### Detection
- DETR: End-to-End Object Detection with Transformers [ECCV2020] [[paper](https://arxiv.org/abs/2005.12872)] [[code](https://github.com/facebookresearch/detr)]
- Deformable DETR: Deformable Transformers for End-to-End Object Detection [ICLR2021] [[paper](https://openreview.net/forum?id=gZ9hCDWe6ke)] [[code](https://github.com/fundamentalvision/Deformable-DETR)]
- End-to-End Object Detection with Adaptive Clustering Transformer [arxiv2020] [[paper](https://arxiv.org/abs/2011.09315)]
- UP-DETR: Unsupervised Pre-training for Object Detection with Transformers [[arxiv2020] [[paper](https://arxiv.org/abs/2011.09094)]
- Rethinking Transformer-based Set Prediction for Object Detection [arxiv2020] [[paper](https://arxiv.org/pdf/2011.10881.pdf)] [[zhihu](https://zhuanlan.zhihu.com/p/326647798)]
- End-to-end Lane Shape Prediction with Transformers [WACV 2021] [[paper](https://arxiv.org/pdf/2011.04233.pdf)] [[code](https://github.com/liuruijin17/LSTR)]
- ViT-FRCNN: Toward Transformer-Based Object Detection [arxiv2020] [[paper](https://arxiv.org/abs/2012.09958)]
- Line Segment Detection Using Transformers [CVPR 2021] [[paper](https://arxiv.org/abs/2101.01909)] [[code](https://github.com/mlpc-ucsd/LETR)]
- Facial Action Unit Detection With Transformers [CVPR 2021] [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Jacob_Facial_Action_Unit_Detection_With_Transformers_CVPR_2021_paper.pdf)] [[code]()]
- Adaptive Image Transformer for One-Shot Object Detection [CVPR 2021] [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Adaptive_Image_Transformer_for_One-Shot_Object_Detection_CVPR_2021_paper.pdf)] [[code]()]
- Self-attention based Text Knowledge Mining for Text Detection [CVPR 2021] [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wan_Self-Attention_Based_Text_Knowledge_Mining_for_Text_Detection_CVPR_2021_paper.pdf)] [[code](https://github.com/CVI-SZU/STKM)]
- Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions [ICCV 2021] [[paper](https://arxiv.org/abs/2102.12122)] [[code](https://github.com/whai362/PVT)]
- Group-Free 3D Object Detection via Transformers [ICCV 2021] [[paper](https://arxiv.org/abs/2104.00678)] [[code](https://github.com/zeliu98/Group-Free-3D)]
- Fast Convergence of DETR with Spatially Modulated Co-Attention [ICCV 2021] [[paper](https://arxiv.org/abs/2101.07448)] [[code](https://github.com/abc403/SMCA-replication)]


<a name="HOI"></a>
### HOI
- End-to-End Human Object Interaction Detection with HOI Transformer [CVPR 2021] [[paper](https://arxiv.org/abs/2103.04503)] [[code](https://github.com/bbepoch/HoiTransformer)]
- HOTR: End-to-End Human-Object Interaction Detection with Transformers [CVPR 2021] [[paper](https://arxiv.org/abs/2104.13682)] [[code](https://github.com/kakaobrain/HOTR)]

<a name="Tracking"></a>
### Tracking
- Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking [CVPR 2021] [[paper](https://arxiv.org/abs/2103.11681)] [[code](https://github.com/594422814/TransformerTrack)]
- TransTrack: Multiple-Object Tracking with Transformer [CVPR 2021] [[paper](https://arxiv.org/abs/2012.15460)] [[code](https://github.com/PeizeSun/TransTrack)]
- Transformer Tracking [CVPR 2021] [[paper](https://arxiv.org/abs/2103.15436)] [[code](https://github.com/chenxin-dlut/TransT)]
- Learning Spatio-Temporal Transformer for Visual Tracking [ICCV 2021] [[paper](https://arxiv.org/abs/2103.17154)] [[code](https://github.com/researchmm/Stark)]

<a name="Segmentation"></a>
### Segmentation
- SETR : Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers [CVPR 2021] [[paper](https://arxiv.org/abs/2012.15840)] [[code](https://github.com/fudan-zvg/SETR)]
- Trans2Seg: Transparent Object Segmentation with Transformer [arxiv2021] [[paper](https://arxiv.org/abs/2101.08461)] [[code](https://github.com/xieenze/Trans2Seg)]
- End-to-End Video Instance Segmentation with Transformers [arxiv2020] [[paper](https://arxiv.org/abs/2011.14503)] [[zhihu](https://zhuanlan.zhihu.com/p/343286325)]
- MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers [CVPR 2021] [[paper](https://arxiv.org/pdf/2012.00759.pdf)] [[official-code](https://github.com/google-research/deeplab2/blob/main/g3doc/projects/max_deeplab.md)] [[unofficial-code](https://github.com/conradry/max-deeplab)]
- Medical Transformer: Gated Axial-Attention for Medical Image Segmentation [arxiv 2020] [[paper](https://arxiv.org/pdf/2102.10662.pdf)] [[code](https://github.com/jeya-maria-jose/Medical-Transformer)]
- SSTVOS: Sparse Spatiotemporal Transformers for Video Object Segmentation [CVPR 2021] [[paper](https://arxiv.org/abs/2101.08833)] [[code](https://github.com/dukebw/SSTVOS)]

<a name="Reid"></a>
### Reid
- Diverse Part Discovery: Occluded Person Re-Identification With Part-Aware Transformer [CVPR 2021] [[paper](https://arxiv.org/abs/2106.04095)] [[code]()]

<a name="Localization"></a>
### Localization
- LoFTR: Detector-Free Local Feature Matching with Transformers [CVPR 2021] [[paper](https://arxiv.org/abs/2104.00680)] [[code](https://github.com/zju3dv/LoFTR)]
- MIST: Multiple Instance Spatial Transformer [CVPR 2021] [[paper](https://arxiv.org/pdf/1811.10725)] [[code](https://github.com/ubc-vision/mist)]

<a name="Generation"></a>
### Generation
- Variational Transformer Networks for Layout Generation [CVPR 2021] [[paper](https://arxiv.org/abs/2104.02416)] [[code](https://github.com/zlinao/Variational-Transformer)]
- TransGAN: Two Transformers Can Make One Strong GAN [[paper](https://arxiv.org/pdf/2102.07074.pdf)] [[code](https://github.com/VITA-Group/TransGAN)]
- Taming Transformers for High-Resolution Image Synthesis [CVPR 2021] [[paper](https://arxiv.org/abs/2012.09841)] [[code](https://github.com/CompVis/taming-transformers)]
- iGPT: Generative Pretraining from Pixels [ICML 2020] [[paper](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)] [[code](https://github.com/openai/image-gpt)]
- Generative Adversarial Transformers [arxiv 2021] [[paper](https://arxiv.org/abs/2103.01209)] [[code](https://github.com/dorarad/gansformer)]
- LayoutTransformer: Scene Layout Generation With Conceptual and Spatial Diversity [CVPR2021] [paper[https://openaccess.thecvf.com/content/CVPR2021/html/Yang_LayoutTransformer_Scene_Layout_Generation_With_Conceptual_and_Spatial_Diversity_CVPR_2021_paper.html]] [[code](https://github.com/davidhalladay/LayoutTransformer)]
- Spatial-Temporal Transformer for Dynamic Scene Graph Generation [ICCV 2021] [[paper](https://arxiv.org/abs/2107.12309)]


<a name="Inpainting"></a>
### Inpainting
- STTN: Learning Joint Spatial-Temporal Transformations for Video Inpainting [ECCV 2020] [[paper](https://arxiv.org/abs/2007.10247)] [[code](https://github.com/researchmm/STTN)]

<a name="Image-enhancement"></a>
### Image enhancement
- Pre-Trained Image Processing Transformer [CVPR 2021] [[paper](https://arxiv.org/abs/2012.00364)]
- TTSR: Learning Texture Transformer Network for Image Super-Resolution [CVPR2020] [[paper](https://arxiv.org/abs/2006.04139)] [[code](https://github.com/researchmm/TTSR)]

<a name="Pose-Estimation"></a>
### Pose Estimation
- Pose Recognition with Cascade Transformers [CVPR 2021] [[paper](https://arxiv.org/abs/2104.06976)] [[code](https://github.com/mlpc-ucsd/PRTR)]
- TransPose: Towards Explainable Human Pose Estimation by Transformer [arxiv 2020] [[paper](https://arxiv.org/abs/2012.14214)] [[code](https://github.com/yangsenius/TransPose)]
- Hand-Transformer: Non-Autoregressive Structured Modeling for 3D Hand Pose Estimation [ECCV 2020] [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700018.pdf)]
- HOT-Net: Non-Autoregressive Transformer for 3D Hand-Object Pose Estimation [ACMMM 2020] [[paper](https://cse.buffalo.edu/~jmeng2/publications/hotnet_mm20)]
- End-to-End Human Pose and Mesh Reconstruction with Transformers [CVPR 2021] [[paper](https://arxiv.org/abs/2012.09760)] [[code](https://github.com/microsoft/MeshTransformer)]
- 3D Human Pose Estimation with Spatial and Temporal Transformers [arxiv 2020] [[paper](https://arxiv.org/pdf/2103.10455.pdf)] [[code](https://github.com/zczcwh/PoseFormer)]
- End-to-End Trainable Multi-Instance Pose Estimation with Transformers [arxiv 2020] [[paper](https://arxiv.org/abs/2103.12115)]


<a name="Face"></a>
### Face
- Robust Facial Expression Recognition with Convolutional Visual Transformers [arxiv 2020] [[paper](https://arxiv.org/abs/2103.16854)]
- Clusformer: A Transformer Based Clustering Approach to Unsupervised Large-Scale Face and Visual Landmark Recognition [CVPR 2021] [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Nguyen_Clusformer_A_Transformer_Based_Clustering_Approach_to_Unsupervised_Large-Scale_Face_CVPR_2021_paper.html)] [[code](https://github.com/lucidrains/TimeSformer-pytorch)]


<a name="Video-Understanding"></a>
### Video Understanding
- Is Space-Time Attention All You Need for Video Understanding? [arxiv 2020] [[paper](https://arxiv.org/abs/2102.05095)] [[code](https://github.com/lucidrains/TimeSformer-pytorch)]
- Temporal-Relational CrossTransformers for Few-Shot Action Recognition [CVPR 2021] [[paper](https://arxiv.org/abs/2101.06184)] [[code](https://github.com/tobyperrett/trx)]
- Self-Supervised Video Hashing via Bidirectional Transformers [CVPR 2021] [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Self-Supervised_Video_Hashing_via_Bidirectional_Transformers_CVPR_2021_paper.html)]
- SSAN: Separable Self-Attention Network for Video Representation Learning [CVPR 2021] [[paper](https://arxiv.org/abs/2105.13033)]

<a name="Depth-Estimation"></a>
### Depth-Estimation
- Adabins：Depth Estimation using Adaptive Bins [CVPR 2021] [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Bhat_AdaBins_Depth_Estimation_Using_Adaptive_Bins_CVPR_2021_paper.pdf)] [[code](https://github.com/shariqfarooq123)]


<a name="Prediction"></a>
### Prediction
- Multimodal Motion Prediction with Stacked Transformers [CVPR 2021] [[paper](https://arxiv.org/pdf/2103.11624.pdf)] [[code](https://github.com/decisionforce/mmTransformer)]
- Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case [[paper](https://arxiv.org/pdf/2001.08317.pdf)]
- Transformer networks for trajectory forecasting [ICPR 2020] [[paper](https://arxiv.org/abs/2003.08111)] [[code](https://github.com/FGiuliari/Trajectory-Transformer)]
- Spatial-Channel Transformer Network for Trajectory Prediction on the Traffic Scenes [arxiv 2021] [[paper](https://arxiv.org/abs/2101.11472)] [[code]()]
- Pedestrian Trajectory Prediction using Context-Augmented Transformer Networks [ICRA 2020] [[paper](https://arxiv.org/abs/2012.01757)] [[code]()]
- Spatio-Temporal Graph Transformer Networks for Pedestrian Trajectory Prediction [ECCV 2020] [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570494.pdf)] [[code](https://github.com/Majiker/STAR)]
- Hierarchical Multi-Scale Gaussian Transformer for Stock Movement Prediction [[paper](https://www.ijcai.org/Proceedings/2020/0640.pdf)]
- Single-Shot Motion Completion with Transformer [arxiv2021] [[paper](https://arxiv.org/abs/2103.00776)] [[code](https://github.com/FuxiCV/SSMCT)]

<a name="NAS"></a>
### NAS
- HR-NAS: Searching Efficient High-Resolution Neural Architectures with Transformers [CVPR 2021] [[paper](https://arxiv.org/abs/2106.06560)] [[code](https://github.com/dingmyu/HR-NAS)]
- AutoFormer: Searching Transformers for Visual Recognition [ICCV 2021] [[paper](https://arxiv.org/abs/2107.00651)] [[code(https://github.com/microsoft/AutoML)]]

<a name="PointCloud"></a>
### PointCloud
- Multi-Modal Fusion Transformer for End-to-End Autonomous Driving [CVPR 2021] [[paper](https://arxiv.org/abs/2104.09224)] [[code](https://github.com/autonomousvision/transfuser)]
- Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos [CVPR 2021] [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Fan_Point_4D_Transformer_Networks_for_Spatio-Temporal_Modeling_in_Point_Cloud_CVPR_2021_paper.html)]

<a name="Fashion"></a>
### Fashion
- Kaleido-BERT：Vision-Language Pre-training on Fashion Domain [CVPR 2021] [[paper](https://arxiv.org/abs/2103.16110)] [[code](https://github.com/mczhuge/Kaleido-BERT)]

<a name="Medical"></a>
### Medical
- Lesion-Aware Transformers for Diabetic Retinopathy Grading [CVPR 2021] [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Lesion-Aware_Transformers_for_Diabetic_Retinopathy_Grading_CVPR_2021_paper.html)]

<a name="Cross-Modal"></a>
## Cross-Modal
- Thinking Fast and Slow: Efficient Text-to-Visual Retrieval with Transformers [CVPR 2021] [[paper](https://arxiv.org/abs/2103.16553)]
- Revamping Cross-Modal Recipe Retrieval with Hierarchical Transformers and Self-supervised Learning [CVPR2021] [[paper](https://www.amazon.science/publications/revamping-cross-modal-recipe-retrieval-with-hierarchical-transformers-and-self-supervised-learning)] [[code](https://github.com/amzn/image-to-recipe-transformers)]
- Topological Planning With Transformers for Vision-and-Language Navigation [CVPR 2021] [[paper](https://arxiv.org/abs/2012.05292)]
- Multi-Stage Aggregated Transformer Network for Temporal Language Localization in Videos [CVPRR 2021] [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Multi-Stage_Aggregated_Transformer_Network_for_Temporal_Language_Localization_in_Videos_CVPR_2021_paper.html)]
- VLN BERT: A Recurrent Vision-and-Language BERT for Navigation [CVPR 2021] [[paper](https://arxiv.org/abs/2011.13922)] [[code](https://github.com/YicongHong/Recurrent-VLN-BERT)]
- Less Is More: ClipBERT for Video-and-Language Learning via Sparse Sampling [CVPR 2021] [[paper](https://arxiv.org/abs/2102.06183)] [[code](https://github.com/jayleicn/ClipBERT)]

# Reference
- Attention 机制详解1，2 [zhihu1](https://zhuanlan.zhihu.com/p/47063917) [zhihu2](https://zhuanlan.zhihu.com/p/47282410)
- [自然语言处理中的自注意力机制（Self-attention Mechanism)](https://www.cnblogs.com/robert-dlut/p/8638283.html)
- Transformer模型原理详解 [[zhihu](https://zhuanlan.zhihu.com/p/44121378)] [[csdn](https://blog.csdn.net/longxinchen_ml/article/details/86533005)]
- [完全解析RNN, Seq2Seq, Attention注意力机制](https://zhuanlan.zhihu.com/p/51383402)
- [Seq2Seq and transformer implementation](https://github.com/bentrevett/pytorch-seq2seq)
- End-To-End Memory Networks [[zhihu](https://zhuanlan.zhihu.com/p/29679742)]
- [Illustrating the key,query,value in attention](https://medium.com/@b.terryjack/deep-learning-the-transformer-9ae5e9c5a190)
- [Transformer in CV](https://towardsdatascience.com/transformer-in-cv-bbdb58bf335e)
- [CVPR2021-Papers-with-Code](https://github.com/amusi/CVPR2021-Papers-with-Code)
- [ICCV2021-Papers-with-Code](https://github.com/amusi/ICCV2021-Papers-with-Code)
# Acknowledgement
Thanks for the awesome survey papers of Transformer.