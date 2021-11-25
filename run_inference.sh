python mmocr/utils/ocr.py \
data/TestA/ \
--det DB_r50 \
    --det-ckpt dbnet/latest.pth \
    --det-batch-size 12 \
--recog SAR \
    --recog-ckpt sar/latest.pth \
    --recog-config configs/textrecog/sar/sar_r31_parallel_decoder_chinese.py \
    --recog-batch-size 40 \
--export results/test_A_DB_SAR/ \
--output results/test_A_DB_SAR/ \
--details