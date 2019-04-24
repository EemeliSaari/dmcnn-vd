# Deep Demosaicing Convolutional Neural Networks

Small project attempting to reproduce and adopt the methods introduced by `Nai-Sheng Syu∗, Yu-Sheng Chen∗, Yung-Yu Chuang` in the paper [Learning Deep Convolutional Networks for
Demosaicing](https://arxiv.org/pdf/1802.03769.pdf).

As of 24.04.2019 the authors have not released the code: [link](http://www.cmlab.csie.ntu.edu.tw/project/Deep-Demosaic/)

## Install

Clone the repo
```
$ git clone https://github.com/EemeliSaari/dmcnn-vd.git
$ cd dmcnn-vd
```

Setup the virtualenv and the ipykernel
```
$ python install virtualenv #optional
$ virtualenv .env
$ source .env/bin/activate #windows: .env\Scripts\activate
(.env)$ pip install -r requirements
(.env)$ pip install ipykernel
(.env)$ ipython kernel install --user --name=dmcnn
```

Install the Pytorch suitable for your machine from official website: [link](https://pytorch.org/get-started/locally/)

Run the notebook
```
(.env)$ jupyter notebook
```

## Notes

> Results are only trained with fraction of the patches used in the original paper causing the model to overfit and not generalize well to other images.

> DMCNN experiment seems to require many more epochs than experimented here.

> Most reproduce problems are 99% likely to be from my code.

## Todo

- Generalize the dataset and patch loader.
- Clean up the training loops to separate utility functions and add training time test set to detect overfitting.
- Maybe experiment with networks that are tradeoff between the shallow and deep.

