# Large-scale Diarization Dataset

本项目分为前端和后端两部分：  
- **前端**：说话人日志记录（Speaker Logging）  
- **后端**：声学模型构建（Acoustic Model）  

---

## 使用说明

### 下载数据

使用 `script` 目录下的下载脚本下载对应数据集，例如：

```
bash script/download_librispeech.sh <your_save_dir>
```
点噪声和漫反射噪声数据可从以下链接下载：https://1drv.ms/u/c/969dad2e7ff5ab41/EcV68xcR9pVHsd3yNWSTzxkBkKvfLwTQsOluZJOnzf1GFA?e=OnfDv5

* 核心环境依赖：
    * faster_whisper==1.1.1 （核心组件，建议所有相关库依据faster_whisper安装）
    * soundfile
    * tqdm
    * torch
    * torchaudio
可以使用项目内提供的 requirements.txt 文件快速配置 Conda 环境：
```
conda create -n diarization_env python=3.x
conda activate diarization_env
pip install -r requirements.txt
```



* Run `run.sh`
关键参数：
```
    $exp_dir                # 生成的wav文件的存放路径      
    librispeech_dir=		# librispeech的路径，确保该路径下存在 SPEAKERS.TXT 文件
    aishell_1_dir=!	        # aishell_1的路径，确保该路径下存在 resource_aishell 文件夹（里面存放speaker.info和lexicon.txt）
    point_noise_path=        # 点噪声数据路径
    diffuse_noise_path=      # 漫反射噪声数据路径
```

* 生成关键设置 config/config.yaml
    * iteration: 枚举迭代数目，即程序会尝试iteration次，每次最多生成一个wav文件，即最后的生成wav文件数小于等于iteration；
    * max_examples: 限定生成最大条数；如果要精确生成wav的个数需要调整本参数，例如设置为100，则固定生成100个wav文件；必须设置得比iteration小，否则会失效
