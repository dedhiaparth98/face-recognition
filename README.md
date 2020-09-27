# Face Recognition Software

I have explained the details about the model structure, dataset, and other implementational details in the blog post here. Here, I will provide the steps for using this repository.

#  Installations

The following steps are tested for ubuntu 20.04 with a python version of 3.8

```
sudo apt update
sudo apt upgrade
sudo snap install cmake
```
Clone the project to any directory and open the terminal in that directory.
We will have to create some directories that will be essential for our storing images and features
```
mkdir data
mkdir features
```

I generally like to create a virtual environment for each of my projects so the next step is optional.

```
virtualenv -p /usr/bin/python3.8 venv
source venv/bin/activate
```
Now we will install all the pip packages required for running this applications. 
```
pip install -r reqirement.txt 
```
If you want to just try out the Browser-Based UI tool or run the notebook, then you can download the pre-trained model weights from [here](https://drive.google.com/file/d/1MegWliwXx2J-xHYX6iETl7hXUtLRk2sC/view?usp=sharing). After extracting the files, your directory should look like this.
## Check you setup
```
.
├── app.py
├── config.py
├── data
├── features
├── logs
│   ├── func
│   ├── model
│   └── scalars
├── notebooks
│   ├── DatagGeneration.ipynb
│   ├── Real-time-prediction.ipynb
│   └── SiameseNetwork-TripletLoss.ipynb
├── README.md
├── reqirement.txt
├── siameseNetwork.py
├── static
│   ├── css
│   └── images
├── templates
│   ├── index.html
│   └── results.html
├── utils.py
```

## Running the browser-based tool

If you have the same set of files and folders in the directory then you can run the following command
```
python app.py
```
The flask app will start and you will be able to collect training data for any new person and generate features for that person and check the real-time recognition. You can add as many people as you want to. The images collected from this tool will be added to the data folder and the corresponding features generated will be stored in the features folder.

_**Note :** If you delete any person's images from the data folder, you need to delete the .pkl file inside the features folder as well. Also, the pickle file will be generated only when you hit submit images in the browser tool._

## Running the Notebooks

You can start the jupyter-lab or jupyter notebook server and check the notebooks folder. 
Notebook [Real-time-prediction.ipynb](https://github.com/dedhiaparth98/face-recognition/blob/master/notebooks/Real-time-prediction.ipynb) can be used for evaluating the model. It's the notebook version of the browser-based tool. However, the prediction in real-time webcam frame is much faster here as the browser sends API calls to the backend and each image frame of the video is send whereas here, it's not necessary.

Other instructions for running this notebook are provided in the notebook itself. However, the data directory is shared between this notebook and the browser-based tool.

## Training from scratch

If you wish to train your model and get your own weights, then you can use [SiameseNetwork-TripletLoss.ipynb](https://github.com/dedhiaparth98/face-recognition/blob/master/notebooks/SiameseNetwork-TripletLoss.ipynb). I had trained the same on colab and kept the lines of code for mounting the drive and other TensorFlow logging. Please refer to the blog post link above for learning more about the training details.

## For Web Developers - I have a question

I have tried using socket connection as well as ajax calls for sending data to the backend while running prediction calls on the images. It was counter-intuitive to know that the socket connection was giving me a slower frame rate than the ajax calls.

The current implementation is with Ajax call but the commented code for the socket is kept in both frontend and backend. **So if you know why is socket slow than ajax ?** then please drop me a message on [twitter](https://twitter.com/Parth_dedhia98) or [linkedin](https://www.linkedin.com/in/parth-dedhia). Thanks in advance !!

## References

1. O. M. Parkhi, A. Vedaldi, A. Zisserman, Deep Face Recognition, British Machine Vision Conference, 2015.
1. Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman, VGGFace2: A dataset for recognising face across pose and age, International Conference on Automatic Face and Gesture Recognition, 2018.
3. F. Schroff, D. Kalenichenko, J. Philbin, FaceNet: A Unified Embedding for Face Recognition and Clustering, CVPR, 2015.
4. G. Koch, R. Zemel, R. Salakhutdinov, Siamese Neural Networks for One-shot Image Recognition, ICML deep learning workshop. Vol. 2. 2015.
5. [https://github.com/rcmalli/keras-vggface](https://github.com/rcmalli/keras-vggface)
6. [https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)
7. [https://medium.com/datadriveninvestor/speed-up-your-image-training-on-google-colab-dc95ea1491cf](https://medium.com/datadriveninvestor/speed-up-your-image-training-on-google-colab-dc95ea1491cf)
