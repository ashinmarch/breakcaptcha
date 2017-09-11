# breakcaptcha
Use tensorflow and deep learning (cnn) to break captcha.image system.
I constructed the 3 layers of conv-maxpooling combinations and 1 1024d fully connected layer. Finally 1024d fully connected layer was mapped to 4 10d outputs, each representing 1 one-hot vector of 1 digit.

The accuracy is ~98%. The prediction is correct only if all 4 digits are predicted correctly.

Usage:
1. Train: *python break_pycaptcha_4d.py -t train*
2. Evaluate: *python break_pycaptcha_4d.py -t eval <NUM>*  (NUM: any integer from [0, 9999])

The captcha system: https://pypi.python.org/pypi/captcha/0.1.1
Source: https://github.com/lepture/captcha
![1234](https://cloud.githubusercontent.com/assets/290496/5213632/95e68768-764b-11e4-862f-d95a8f776cdd.png)
