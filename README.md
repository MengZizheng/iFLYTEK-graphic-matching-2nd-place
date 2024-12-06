# 2024 iFLYTEK 大模型图文匹配识别挑战赛 - 第2名方案

## 赛题地址
[大模型图文匹配识别挑战赛](https://challenge.xfyun.cn/topic/info?type=graphic-matching&option=ssgy)

## 赛题背景
图文匹配技术在多个领域有着广泛的应用，如根据用户的兴趣和搜索历史推荐相关图文内容。在大模型应用场景中，多模态学习是一个重要的研究方向，它涉及到将不同模态的信息（如文本、图像、音频等）进行有效整合，以提高模型的理解和表达能力。图文匹配识别作为多模态学习中的一个关键问题，要求模型能够理解图像内容并将其与相应的文本描述进行匹配，

## 赛题任务
参赛者需要使用主办方提供的[数据集](https://challenge.xfyun.cn/topic/info?type=graphic-matching&option=stsj)，该数据集包含大量图像及其对应的文本描述。参赛者需要设计一个能够处理图文匹配任务的模型，可以是传统的机器学习方法，也可以是深度学习模型。最终参赛选手需要在测试集上完成图文匹配的操作。

## 方案概述
本方案基于CLIP模型，结合LoRA（Low-Rank Adaptation）技术进行模型微调，旨在提高图文匹配任务中的效果。通过使用CLIP模型对图像和文本进行嵌入向量化，并应用匈牙利算法（Hungarian Algorithm）进行匹配优化，模型能够更好地处理图像和文本之间的语义关系。

### 主要技术方案
1. **CLIP模型微调**：使用OFA-Sys团队提供的`chinese-clip-vit-huge-patch14`预训练模型，并通过LoRA技术进行微调，只优化模型中的关键部分，减少训练成本和时间。
2. **LoRA技术**：采用LoRA技术优化CLIP模型的查询、键、值等参数，提升多模态学习效果。
3. **数据预处理**：处理图像和文本数据，使其适应模型的输入要求，并进行批量化处理。
4. **图文匹配优化**：通过CLIP模型进行图像和文本向量化，并使用匈牙利算法对相似度矩阵进行优化，确保图文匹配的准确性。
5. **模型训练**：利用训练集进行模型的微调，采用AdamW优化器和交叉熵损失函数，训练过程中记录日志和评估模型性能。

## 环境要求
- Python 3.7及以上版本
- 依赖库：`transformers`，`torch`，`peft`，`datasets`，`scipy`，`pandas`，`numpy`，`sklearn`，`PIL`，`matplotlib`
- GPU（推荐使用NVIDIA A100或V100）

## 使用方法

### 1. 数据加载与预处理
首先读取训练集和测试集数据，并对图像进行路径拼接处理，准备训练数据。

```python
df_train = pd.read_csv('..xfdata/dataset/train.csv')
df_train["image_path"] = df_train["image_name"].apply(lambda x: os.path.join(img_path, x))
train_dataset = Dataset.from_pandas(df_train)
```

### 2. 模型加载与LoRA设置
使用`ChineseCLIPModel`加载CLIP模型，并通过LoRA技术进行微调，仅优化特定层，提升模型训练效率。

```python
model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-huge-patch14", cache_dir="../user_data/").to(device)
processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-huge-patch14", cache_dir="../user_data/")
```

### 3. 训练与评估
使用交叉熵损失函数进行训练，并在每个epoch后评估模型的性能，记录训练过程中的损失和准确率。

```python
train_loss, train_acc = train_epoch(lora_model, train_dataloader, optimizer, device)
valid_loss, valid_acc = evaluate_epoch(lora_model, valid_dataloader, device)
```

### 4. 图文匹配与优化
通过CLIP模型将图像和文本转化为向量，并使用匈牙利算法对图像和文本之间的匹配进行优化，最终生成匹配结果。

```python
similar_matrix = torch.mm(text_embeddings, image_embeddings.T).detach().cpu().numpy()
row_ind, col_ind = linear_sum_assignment(similar_matrix, maximize=True)
```

### 5. 提交结果
将最终的匹配结果保存为`result.csv`文件，按要求格式提交。

```python
df_submit = pd.read_csv('../xfdata/sample_submit.csv')
df_submit['image_name'] = image_name_all
df_submit.to_csv('../prediction_result/result.csv', index=False)
```

## 7. 算力平台
为了高效训练模型，我们使用了 [onethingai](https://onethingai.com/invitation?code=wGZHFckZ) 提供的算力平台。该平台提供了强大的GPU资源，使我们能够在较短的时间内完成模型训练和微调。

## 8. 贡献者
团队名称：小老正  
成员：[孟子正]