## Дополнительные зависимости для giga-tts

1. ### Если нужно считать wer, cer
```bash
pip install -e contrib/pyannote-audio --break-system-packages
pip install -e contrib/GigaAM[longform] --break-system-packages
```

2. ### Если нужно синтезировать вавки с помощью gigaam-codec
```bash
pip install -e contrib/gigaam-codec[gigaam,loc_attn,fsq] --break-system-packages
pip install --upgrade 'protobuf==4.25.6' 'tensorboardX<2.3' --break-system-packages
```

3. ### Если нужно считать UTMOS
```bash
pip install https://github.com/vBaiCai/python-pesq/archive/master.zip  --break-system-packages
pip install git+https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics.git --break-system-packages
```

4. ### Если нужно считать SSIM с помощью wavlm (по возможности избегайте этот шаг)
```bash
pip install \
    s3prl@git+https://github.com/s3prl/s3prl.git@7ab62aaf2606d83da6c71ee74e7d16e0979edbc3#egg=s3prl \
    --default-timeout 1000 --break-system-packages
```
