# breakcaptcha
Use tensorflow and deep learning (cnn) to break captcha.image system.
I constructed the 3 layers of conv-maxpooling combinations and 1 1024d fully connected layer. Finally 1024d fully connected layer was mapped to 4 10d outputs, each representing 1 one-hot vector of 1 digit.

Usage:
1. Train
    python break_pycaptcha_4d.py -t train
2. Evaluate
    python break_pycaptcha_4d.py -t eval <NUM>
    <NUM>: any integer from [0, 9999]
