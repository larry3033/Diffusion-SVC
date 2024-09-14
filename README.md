

# Diffusion-SVC


~~此仓库是[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)仓库的扩散部分的单独存放。可单独训练和推理。~~（V2不是啦）<br>
欢迎来到SVC试验田
***
**最近更新：使用本仓库的naive模型和浅扩散模型搭配可以用极低训练成本达到比单纯扩散模型更好的效果，强力推荐。但是小网络的naive模型泛化能力较弱，在小数据集上可能会有音域问题，这个时候naive模型微调不能训练太多步数(这会让底模退化)，前级也可以考虑更换为无限音域的ddsp模型。**
<br>效果和介绍见[[介绍视频(暂未完成)]]()
**欢迎加群交流讨论：882426004**
![Diagram](doc/diagram.jpg)
## 0.简介
Diffusion-SVC-V2 是由V1改进得来的。可单独训练和推理。

相比于比较著名的 [Diff-SVC](https://github.com/prophesier/diff-svc), 本项目的显存占用少得多，训练和推理速度更快，并针对浅扩散和实时用途有专门优化。可以在较强的GPU上实时推理。配合本项目的naive模型进行浅扩散，即使是较弱的GPU也可以实时生成质量优秀的音频。

如果训练数据和输入源的质量都非常高，Diffusion-SVC可能拥有最好的转换效果。

本项目可以很容易的级联在别的声学模型之后进行浅扩散，以改善最终的输出效果或降低性能占用。例如在[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)和本项目的**naive模型**后级联Diffusion-SVC，可进一步减少需要的扩散步数并得到高质量的输出。

除此之外，本项目还可以单独训练浅扩散所需的降噪步数而不训练完整的从高斯噪声开始的降噪过程，这可以提高训练速度并改善质量，更多信息见下文。

免责声明：请确保仅使用**合法获得的授权数据**训练 Diffusion-SVC 模型，不要将这些模型及其合成的任何音频用于非法目的。 本库作者不对因使用这些模型检查点和音频而造成的任何侵权，诈骗等违法行为负责。

## 1. 安装依赖
1. 安装PyTorch：我们推荐从 **[PyTorch 官方网站 ](https://pytorch.org/)** 下载 PyTorch.

2. 安装依赖
```bash
pip install -r requirements.txt
#对于Windows系统，你可能需要预先安装C++生成工具 
```
经过测试，项目可以在Python3.8+cu118或者Python+cu121正常运行
## 2. 配置预训练模型
- **(必要操作)** 下载预训练特征提取编码器并将其放到 `pretrain` 文件夹。 [ContentVec](https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr) 或者由[svc-develop-team](https://github.com/svc-develop-team/contentvec_test) 提供的优化版ContentVec并将其放到 `pretrain/contentvec` 文件夹。
  - 注意：也可以使用别的特征提取，但仍然优先推荐ContentVec。支持的所有特征提取见`tools/tools.py`中的`Units_Encoder`类。
- **(必要操作)** 从 [DiffSinger 社区声码器项目](https://openvpi.github.io/vocoders) 或者[Kouon-Vocoder 社区声码器项目](https://github.com/Kouon-Vocoder-Project)下载预训练声码器，并解压至 `pretrain/` 文件夹。
  -  注意：你应当下载名称中带有`nsf_hifigan`的压缩文件，而非`nsf_hifigan_finetune`。
- **(可选操作)**  从 [RMVPE](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip) 下载预训练rmvpe提取器并解压至 `pretrain/` 文件夹

## 3. 预处理

### 1. 配置训练数据集和验证数据集

#### 1.1 手动配置：

将所有的训练集数据 (.wav 格式音频切片) 放到 `data/train/audio`,也可以是配置文件中指定的文件夹如`xxxx/yyyy/audio`。

将所有的验证集数据 (.wav 格式音频切片) 放到 `data/val/audio`,也可以是配置文件中指定的文件夹如`aaaa/bbbb/audio`。

在配置文件中的位置如下
```bash
data:
  ...
  train_path: op_data/train # Create a folder named "audio" under this path and put the audio clip in it
  valid_path: op_data/val # Create a folder named "audio" under this path and put the audio clip in it
  ...
```

#### 1.2文件夹结构目录展示：
**注意：说话人id必须从1开始，不能从0开始；如果只有一个说话人则该说话人id必须为1**<br>
**所有说话人目录下不应该有任何子目录**
- 目录结构：

```
data
├─ train
│    ├─ audio
│    │    ├─ 1
│    │    │   ├─ aaa.wav
│    │    │   ├─ bbb.wav
│    │    │   └─ ....wav
│    │    ├─ 2
│    │    │   ├─ ccc.wav
│    │    │   ├─ ddd.wav
│    │    │   └─ ....wav
│    │    └─ ...
|
├─ val
|    ├─ audio
│    │    ├─ 1
│    │    │   ├─ eee.wav
│    │    │   ├─ fff.wav
│    │    │   └─ ....wav
│    │    ├─ 2
│    │    │   ├─ ggg.wav
│    │    │   ├─ hhh.wav
│    │    │   └─ ....wav
│    │    └─ ...
```

#### 2. 正式预处理
```bash
python preprocess.py -c <your_config_path/config_name.yaml>
```
您可以在预处理之前修改配置文件

#### 3. 备注：
1. 请保持所有音频切片的采样率与 yaml 配置文件中的采样率一致！（推荐使用[fap](https://github.com/fishaudio/audio-preprocess)进行重采样的等前处理）但是实际上你只能用44100，因为声码器的采样率。

2. 长音频切成小段可以加快训练速度，但所有音频切片的时长不应少于 2 秒。如果音频切片太多，则需要较大的内存（或者显存），配置文件中 `cache_fp16` 改为 ture 缓解或者将 `cache_all_data` 选项设置为 false 解决此问题。（但会对硬盘有极大的损耗，机械硬盘请勿开启）

3. 验证集的音频切片不要放太多，不然验证过程会很慢。

4. 如果您的数据集质量不是很高，请参阅下方的**f0提取器**板块，选择合适的f0预测器并在配置文件 `f0_extractor` 处修改。

5. 配置文件中的 ‘n_spk’ 参数将控制是否训练多说话人模型。如果您要训练**多说话人**模型，为了对说话人进行编号，所有音频文件夹的名称必须是**不大于 ‘n_spk’ 的正整数**

## 4. 训练

### 1. 不使用预训练数据进行训练：
```bash
python train.py -c <your_config_path/config_name.yaml>
```

### 2. 预训练模型：
- **我们强烈建议使用预训练模型进行微调，这将比直接训练容易和节省的多，并能达到比小数据集更高的上限。**

- **注意，在底模上微调需要使用和底模一样的编码器，如同为ContentVec，对别的编码器也是同理，还要注意模型的网络大小等参数相同。**
```bash
#网络参数在配置文件的这里
model:
    ...
    n_layers: 3
    n_chans: 256
    ...
```
****
#### ！！！！！！！！！推荐训练带Reflow的模型！！！！！！！！！
数据过差的话可能会炸
****



### 3. 使用预训练数据（底模）进行训练：
1. 欢迎PR训练的底模 (请使用授权同意开源的数据集进行训练)。
2. 预训练模型需要特别注意使用的是相同编码器的模型。
3. 将名为`model_0.pt`的预训练模型, 放到`config.yaml`里面 "expdir: exp/*****" 参数指定的模型导出文件夹内, 没有就新建一个, 程序会自动加载该文件夹下的预训练模型。
4. 同不使用预训练数据进行训练一样，启动训练。

### 4.我们提供的预训练模型
看todo

### 4.1 不同配置文件的特性解释

## 5. 可视化
```bash
# 使用tensorboard检查训练状态
tensorboard --logdir=exp
```
第一次验证后，在 TensorBoard 中可以看到合成后的测试音频。

## 6. 非实时推理
```bash
python main.py -i <input.wav> -model <model_ckpt.pt> -o <output.wav> -k <keychange> -id <speaker_id> -speedup <speedup> -method <method> -kstep <kstep> -nmodel <nmodel> -pe <f0_extractor>
```
必须的参数：<br>
`-model`模型路径

`-k` 变调,整数类型，数字为半音

`-id`全局使用的说话人id。多说话人必须，默认1

`-i` 输入音频路径

`-o` 输出音频路径

`-th` 响应阈值

`-inferstep` reflow采样步数

`-method` 可选`rk4` `euler`


非必要参数：<br>

`-pe` 使用的F0提取器可选 `crepe` `parselmouth` `dio` `harvest` `rmvpe`  `fcpe`   默认 `crepe`  

`-d`使用的设备，可选`cpu` `cuda`，默认自动

`-tstart` reflow部分处理多少前级输出结果


已经被抛弃的参数和特性：<br>
~~`-speedup`为加速倍速~~
~~`-method`为`pndm``ddim`,`unipc`或`dpm-solver`~~
~~`-kstep`浅扩散步数~~
~~如果`-kstep`不为空，则以输入源的 mel 进行浅扩散，若`-kstep`为空，则进行完整深度的高斯扩散。~~
~~`-nmodel`(可选，需要单独训练)是naive模型的路径，用来提供一个大致的mel给扩散模型进行k_step深度的浅扩散，其参数需要与主模型匹配。~~




## 8. 实时推理
推荐使用本仓库自带的GUI进行实时推理
```bash
python gui_realtime_reflow.py
```


## 9. 兼容性
### 9.1. Units编码器
|                               | Diffusion-SVC | [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) | [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) |
|-------------------------------|---------------|------------------------------------------------|----------------------------------------------------------------|
| ContentVec                    | √             | √                                              | √                                                              |
| HubertSoft                    | √             | √                                              | √                                                              |
| Hubert(Base,Large)            | √             | √                                              | ×                                                              |
| CNHubert(Base,Large)          | √             | √                                              | √*                                                             |
| CNHubertSoft                  | √             | √                                              | ×                                                              |
| Wav2Vec2-xlsr-53-espeak-cv-ft | √*            | ×                                              | ×                                                              |
| DPHubert                      | ×             | ×                                              | √                                                              |
| Whisper-PPG                   | ×             | ×                                              | √*                                                             |
| WavLM(Base,Large)             | ×             | ×                                              | √*                                                             |

## 10. Colab
~~暂时不可用~~

## 11.Onnx导出
~~暂时不可用~~

## 12.Todo
|                               | 完成情况 | 
|-------------------------------|---------------|
| 易于用户使用的Gradio界面        | ×             | 
| ONNX导出                       | ×             | 
| 更好的语义编码器               | ×*             |
| 更好的预处理               | √（实验性）     |
| 预训练模型           | √（实验性）     |

## 13.f0提取器
| 预测器	| 优点 |	缺点 |
|-------------------------------|---------------|---------------|
| pm  |	速度快，占用低 |	容易出现哑音|
| crepe |	基本不会出现哑音 |	显存占用高，自带均值滤波，因此可能会出现跑调|
| dio |	-|	可能跑调 |
| harvest |	低音部分有更好表现	| 其他音域就不如别的算法 |
| rmvpe |	六边形战士，目前最完美的预测器	| 几乎没有缺点（极端长低音可能会出错）|
| fcpe	| 吃土自研，目前最快的预测器，且有不输 crepe 的准确度 |	-|

## 感谢
* [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)
* [soft-vc](https://github.com/bshall/soft-vc)
* [diff-SVC](https://github.com/prophesier/diff-SVC)
* [DiffSinger (OpenVPI version)](https://github.com/openvpi/DiffSinger)

## 感谢所有贡献者作出的努力
<a href="https://github.com/CNChTu/Diffusion-SVC/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=CNChTu/Diffusion-SVC" />
</a>
