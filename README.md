## Replay Memory as An Empirical MDP: Combining Conservative Estimation with Experience Replay

![overview](https://github.com/initial-h/ceer/blob/main/pic/overview.png)

## Overview

- PyTorch implementation of Conservative Estimation with Experience Replay ([CEER](https://openreview.net/forum?id=SjzFVSJUt8S)). 

- Method is tested on [Sokoban](https://github.com/mpSchrader/gym-sokoban), [Minigrid](https://github.com/Farama-Foundation/Minigrid) and [MinAtar](https://github.com/kenjyoung/MinAtar) environments.

## Installation
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```
- My Python version is 3.7.11. CUDA version is 11.4.

## Running Experiments

```
python main.py
```
- Modify `atari_name_list` in `ceer/arguments.py` for different environments.

- For example, `'atari_name_list': ['Sokoban-Push_5x5_1_120']`.

- Other parameters like `sample_method_para # alpha`,`policy_loss_para # lambda` are also in `ceer/arguments.py`.
  
 ## Bibtex
```
@inproceedings{
zhang2023replay,
title={Replay Memory as An Empirical {MDP}: Combining Conservative Estimation with Experience Replay},
author={Hongming Zhang and Chenjun Xiao and Han Wang and Jun Jin and Bo Xu and Martin M{\"u}ller},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=SjzFVSJUt8S}
}
```
 
## Acknowledgments

- Awesome Environments used for testing:

  Sokoban: https://github.com/mpSchrader/gym-sokoban

  Minigrid: https://github.com/Farama-Foundation/Minigrid
  
  MinAtar: https://github.com/kenjyoung/MinAtar


- Some baselines can be found in following works:
 
  TER: https://openreview.net/forum?id=OXRZeMmOI7a
  
  Dreamerv2: https://github.com/RajGhugare19/dreamerv2
  
  Tianshou: https://github.com/thu-ml/tianshou
