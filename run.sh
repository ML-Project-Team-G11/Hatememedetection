ssh 10.127.30.12 /home/ridwan.salahuddeen/.conda/envs/clipenv/bin/python \
    /home/ridwan.salahuddeen/Documents/ML701/Project/Hatememe_project/main.py \
    --fusion_method=align \
    --activation=gelu \
    --dropout_prob=0.3 \
    --learning_rate=0.001 \
    --batch_size=64 \
    --add_memotion=true \
    --add_linear_image_layers=true \
    --add_linear_text_layers=true \
    --train_image_base_model=false \
    --train_text_base_model=false 