docker run --shm-size=8g -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 \
--cpus 20 \
--name aiclub_aic21_duyt_mmocr \
-v /home/duyt/AIC/mmocr:/mmocr \
-v /home/duyt/AIC/data:/mmocr/data \
-v /home/duyt/AIC/results:/mmocr/results \
aiclub_aic21_duyt_mmocr