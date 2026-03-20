# onevision_next
conda create -n onevision_next python=3.10 #建onevision_next這個環境
conda activate onevision_next
## Environment
- Python 3.10
- torch 2.7.0+cu128
- torchvision 0.22.0+cu128
- torchaudio 2.7.0+cu128
- transformers 4.55.4
- accelerate 1.12.0
- bitsandbytes 0.49.1
- pandas 2.3.3
- numpy 1.26.4
- scikit-learn 1.7.2

## Install
```bash
conda create -n onevision_next python=3.10
conda activate onevision_next

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate bitsandbytes pandas numpy scikit-learn pillow tqdm sentencepiece
