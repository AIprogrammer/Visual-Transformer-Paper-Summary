# Milestones
## Attention
- Recurrent Models of Visual Attention [2014 deepmind NIPS]
- Neural Machine Translation by Jointly Learning to Align and Translate [ICLR 2015]
- 
## NLP
- Sequence to Sequence Learning with Neural Networks [NIPS 2014] [[paper](https://arxiv.org/abs/1409.3215)] [[code](https://github.com/bentrevett/pytorch-seq2seq)]
- End-To-End Memory Networks [NIPS 2015] [[paper](https://arxiv.org/abs/1503.08895)] [[code](https://github.com/nmhkahn/MemN2N-pytorch)]
- Attention is all you need [NIPS 2017] [[paper](https://arxiv.org/abs/1706.03762)] [[code]()]
- **B**idirectional **E**ncoder **R**epresentations from **T**ransformers: BERT [[paper]()] [[code](https://huggingface.co/transformers/)] [[pretrained-models](https://huggingface.co/transformers/pretrained_models.html)]
- Reformer: The Efficient Transformer [ICLR2020] [[paper](https://arxiv.org/abs/2001.04451)] [[code](https://github.com/lucidrains/reformer-pytorch)]
- Linformer: Self-Attention with Linear Complexity [AAAI2020] [[paper](https://arxiv.org/abs/2006.04768)] [[code](https://github.com/lucidrains/linformer)]

## CV
### Classification
#### Papers and Codes
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale [VIT] [ICLR 2021] [[paper](https://arxiv.org/abs/2010.11929)] [[code](https://github.com/lucidrains/vit-pytorch)]
    - Trained with extra private data: do not generalized well when trained on insufficient amounts of data
- DeiT: Data-efficient Image Transformers [arxiv2021] [[paper](https://arxiv.org/abs/2012.12877)] [[code](https://github.com/facebookresearch/deit)]
    - Token-based strategy and build upon VIT and convolutional models
    - 
#### Interesting Repos
- [Convolutional Cifar10](https://github.com/kuangliu/pytorch-cifar/blob/master/main.py)
- [vision-transformers-cifar10](https://github.com/kentaroy47/vision-transformers-cifar10)
    - Found that performance was worse than simple resnet18
    - The influence of hyper-parameters: dim of vit, etc.
- [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
    - Using pretrained weights can get better results
### Detection
- DETR: End-to-End Object Detection with Transformers [ECCV2020] [[paper](https://arxiv.org/abs/2005.12872)] [[code](https://github.com/facebookresearch/detr)]
- Deformable DETR: Deformable Transformers for End-to-End Object Detection [ICLR2021] [[paper](https://openreview.net/forum?id=gZ9hCDWe6ke)] [[code](https://github.com/fundamentalvision/Deformable-DETR)]
### Segmentation
- SETR : Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers [arxiv2021] [[paper](https://arxiv.org/abs/2012.15840)] [[code](https://github.com/fudan-zvg/SETR)]
- Trans2Seg: Transparent Object Segmentation with Transformer [arxiv2021] [[paper](https://arxiv.org/abs/2101.08461)] [[code](https://github.com/xieenze/Trans2Seg)]
### Generation
- TransGAN: Two Transformers Can Make One Strong GAN [[paper](https://arxiv.org/pdf/2102.07074.pdf)] [[code](https://github.com/VITA-Group/TransGAN)]
- Taming Transformers for High-Resolution Image Synthesis [[paper](https://arxiv.org/abs/2012.09841)] [[code](https://github.com/CompVis/taming-transformers)]
### Prediction
- Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case [[paper](https://arxiv.org/pdf/2001.08317.pdf)]
- Transformer Networks for Trajectory Forecasting [[paper](https://arxiv.org/pdf/2003.08111.pdf)]
- Hierarchical Multi-Scale Gaussian Transformer for Stock Movement Prediction [[paper](https://www.ijcai.org/Proceedings/2020/0640.pdf)]
# Reference
- Attention 机制详解1，2 [zhihu1](https://zhuanlan.zhihu.com/p/47063917) [zhihu2](https://zhuanlan.zhihu.com/p/47282410)
- [自然语言处理中的自注意力机制（Self-attention Mechanism)](https://www.cnblogs.com/robert-dlut/p/8638283.html)
- Transformer模型原理详解 [[zhihu](https://zhuanlan.zhihu.com/p/44121378)] [[csdn](https://blog.csdn.net/longxinchen_ml/article/details/86533005)]
- [完全解析RNN, Seq2Seq, Attention注意力机制](https://zhuanlan.zhihu.com/p/51383402)
- [Seq2Seq and transformer implementation](https://github.com/bentrevett/pytorch-seq2seq)
- End-To-End Memory Networks [[zhihu](https://zhuanlan.zhihu.com/p/29679742)]
- [Illustrating the key,query,value in attention](https://medium.com/@b.terryjack/deep-learning-the-transformer-9ae5e9c5a190) 
