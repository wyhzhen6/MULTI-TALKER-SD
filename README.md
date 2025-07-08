# Large-scale-diarization-dataset
This project is divided into front-end and back-end. The front-end is speaker logging, and the back-end builds an acoustic model.
<<<<<<< HEAD

=======
>>>>>>> 601c6745ccd15d80625b76ace7841d625e48178d
### Use
* Use `script/download_*.sh` to download the corresponding folder, for example:
```
bash script/download_librispeech.sh <your_save_dir>
```

<<<<<<< HEAD
=======
* 核心环境依赖：
    * faster_whisper==1.1.1 （核心组件，建议所有相关库依据faster_whisper安装）
    * soundfile
    * tqdm
    * torch
    * torchaudio

>>>>>>> 601c6745ccd15d80625b76ace7841d625e48178d
* Run `run.sh`
关键参数：
```
    $exp_dir                # 生成的wav文件的存放路径      
    librispeech_dir=		# librispeech的路径，确保该路径下存在 SPEAKERS.TXT 文件
    aishell_1_dir=!	        # aishell_1的路径，确保该路径下存在 resource_aishell 文件夹（里面存放speaker.info和lexicon.txt）
```

* 生成关键设置 config/config.yaml
    * iteration: 枚举迭代数目，即程序会尝试iteration次，每次最多生成一个wav文件，即最后的生成wav文件数小于等于iteration；
<<<<<<< HEAD
    * max_examples: 限定生成最大条数；如果要精确生成wav的个数需要调整本参数，例如设置为100，则固定生成100个wav文件；必须设置得比iteration小，否则会失效
=======
    * max_examples: 限定生成最大条数；如果要精确生成wav的个数需要调整本参数，例如设置为100，则固定生成100个wav文件；必须设置得比iteration小，否则会失效

>>>>>>> 601c6745ccd15d80625b76ace7841d625e48178d
