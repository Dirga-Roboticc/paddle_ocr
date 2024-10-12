cara buat dataset
-download dan buka PPOCRLabel
-buat dataset
cara latih
-edit file ./configs/my_config/ch_PP-OCRv4_det_cml.yml ubah path dataset Train
-python tools/train.py --config ./configs/my_config/ch_PP-OCRv4_det_cml.yml
-python tools/eval.py --config configs/my_config/ch_PP-OCRv4_det_cml.yml -o Global.checkpoints=./output/ch_PP-OCRv4/latest.pdparams
