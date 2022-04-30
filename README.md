# Style_Transfer

Implementation of the paper "Content preserving text generation with attribute controls", Logeswaran et al., which talks 'bout a backtranslation based style transfer network.


## Deployment

First of all, place all the files/folders present in this repository inside a new directory named "my_CPTG", else you will get an alert mentioning the same.

Next, re-install torch and torchtext versions 1.8.0 and 0.9.0 [with(out) cudnn as nccessary] by this command
```bash
$ pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 torchtext==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```
Choose the dataset to train on by manipulating `dataset_name` variable in `main.py` and `eval.py`files by names yelp/amazon/imdb.

To train the model run

```bash
  $ python3 main.py
```
To test the model run
```bash
  $ python3 eval.py
```


## Acknowledgements

 - [Thanks to this nice repo](https://github.com/hwijeen/CPTG)
 - [Paper link](https://arxiv.org/pdf/1811.01135v1.pdf)
 
