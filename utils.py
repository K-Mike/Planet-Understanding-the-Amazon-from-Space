from sklearn.preprocessing import LabelBinarizer
import numpy as np
import glob
from tqdm import tqdm
import pandas as pd
from keras import backend as K
from keras.preprocessing import image
import os
from sklearn.metrics import fbeta_score, precision_score, make_scorer, average_precision_score
import scipy
from skimage.color import convert_colorspace
from skimage.color import rgb2ycbcr, rgb2gray, rgb2hed, rgb2ypbpr, rgb2ycbcr, rgb2lab, rgb2luv


class TagsTransform(object):

    def __init__(self, df):
        labels = ' '.join(df.tags).split()
        self.labels_u = list(set(labels))
        self.labels_u.sort()

        self.lb = LabelBinarizer()
        self.lb.fit(self.labels_u)

    def tags_to_vec(self, tags):

        output_matrix = np.zeros(tags.shape + self.lb.classes_.shape, dtype=int)
        for i, tag in enumerate(tags):
            tag = list(set(tag.split()))
            output_matrix[i] = self.lb.transform(tag).sum(axis=0)

        return output_matrix

    def vec_to_tags(self, y_pred, best_threshold=None):
        if best_threshold is None:
            best_threshold = [0.2]*17

        y_pred_thred = (y_pred.astype(float) > best_threshold).astype(int)
        y_tags = [' '.join(self.lb.classes_[np.where(row == 1)[0]]) for row in y_pred_thred]

        return y_tags


def predict_test(predictor, directory, tagsTransform, batch_size=1000, image_shape=(256, 256, 3),
                 scale=1./255, best_threshold=None):
    files = glob.glob(directory + '*.jpg')

    files_N = len(files)
    pbar = tqdm(total=files_N)

    batch_x = []
    batch_names = []
    result = pd.DataFrame()
    for i, img_path in enumerate(files):

        img = image.load_img(img_path, target_size=image_shape[:-1])
        x = image.img_to_array(img)
        x *= scale
        x = np.expand_dims(x, axis=0)
        batch_x.append(x)

        fname = os.path.basename(img_path)[:-4]
        batch_names.append(fname)

        if (i % batch_size == 0 or i == files_N - 1) and i != 0:
            batch_x = np.vstack(batch_x)
            batch_y = predictor.predict(batch_x)

            # batch_y = (batch_y > best_threshold).astype(int)[0]
            batch_tags = tagsTransform.vec_to_tags(batch_y, best_threshold=best_threshold)

            data = {'image_name': batch_names, 'tags': batch_tags}
            batch_result = pd.DataFrame(data)
            result = result.append(batch_result, ignore_index=True)

            batch_x = []
            batch_names = []

        pbar.update(1)

    return result


def fbeta(true_label, prediction):
    return fbeta_score(true_label, prediction, beta=2, average='samples')


def get_optimal_threshhold(true_label, prediction, iterations=100):

    best_threshhold = [0.2]*17
    for t in range(17):
        best_fbeta = 0
        temp_threshhold = [0.2]*17
        for i in range(iterations):
            temp_value = i / float(iterations)
            temp_threshhold[t] = temp_value
            temp_fbeta = fbeta(true_label, prediction > temp_threshhold)
            if temp_fbeta > best_fbeta:
                best_fbeta = temp_fbeta
                best_threshhold[t] = temp_value

    return best_threshhold


def get_stat_features(img):
    fromspace = 'RGB'
    tospaces = ['YIQ', 'YUV', 'HSV', 'RGB CIE', 'YPbPr', 'XYZ', 'YCbCr', 'gray', 'HED', 'lab', 'LUV']

    feature_vec = []
    for tospace in tospaces:

        if tospace == 'gray':
            img_converted = rgb2gray(img)
        elif tospace == 'HED':
            img_converted = rgb2hed(img)
        elif tospace == 'lab':
            # there can be parameters
            img_converted = rgb2lab(img)
        elif tospace == 'LUV':
            img_converted = rgb2luv(img)
        elif tospace == 'YPbPr':
            img_converted = rgb2ypbpr(img)
        elif tospace == 'YCbCr':
            img_converted = rgb2ycbcr(img)
        else:
            img_converted = convert_colorspace(img, fromspace, tospace)

        arrays = []
        arrays.append(img_converted)

        if tospace != 'gray':
            arrays.extend([img_converted[:,:,i] for i in range(img_converted.shape[2])])

        for arr in arrays:
            description = scipy.stats.describe(arr, axis=None)
            #     add minmax
            feature_vec.append(description[1][0])
            feature_vec.append(description[1][1])
            #     add mean
            feature_vec.append(description[2])
            #     add variance
            feature_vec.append(description[3])
            #     add skewness
            feature_vec.append(description[4])
            #     add kurtosis
            feature_vec.append(description[5])

    feature_vec = np.array(feature_vec)

    return feature_vec

if __name__ == '__main__':
    pass