from keras_multy_lable import Iterator
from keras_multy_lable import load_img
from keras_multy_lable import img_to_array
from sklearn.preprocessing import LabelBinarizer
from keras import backend as K
import numpy as np
import os
from skimage import io
from utils import get_stat_features


class DirectoryGenerator(Iterator):

    def __init__(self, directory, Y_train_data, image_data_generator, target_size=(256, 256), batch_size=32,
                 shuffle=True, seed=None, color_mode='rgb', data_format=None):

        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if data_format is None:
            data_format = K.image_data_format()
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        # init Y_train
        self.Y_train_data = Y_train_data
        labels = ' '.join(Y_train_data.tags).split()
        labels_u = list(set(labels))
        labels_u.sort()
        labelBinarizer = LabelBinarizer()
        labelBinarizer.fit(labels_u)
        self.labelBinarizer = labelBinarizer

        self.samples = self.Y_train_data.shape[0]
        self.num_class = len(labels_u)
        print('Found %d images belonging to %d classes.' % (self.samples, self.num_class))

        super(DirectoryGenerator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _tag_to_vec(self, tags):
        return self.labelBinarizer.transform(tags.split()).sum(axis=0)

    def next(self):
        index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        batch_y = np.zeros((current_batch_size,) + self.labelBinarizer.classes_.shape)

        grayscale = self.color_mode == 'grayscale'
        for i, j in enumerate(index_array):
            fname = self.Y_train_data.image_name.ix[j] + '.jpg'
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y[i] = self._tag_to_vec(self.Y_train_data.tags.ix[j])

        return batch_x, batch_y


class DirectoryGeneratorStat(Iterator):

    def __init__(self, directory, Y_train_data, image_data_generator, target_size=(256, 256), batch_size=32,
                 shuffle=True, seed=None, color_mode='rgb', data_format=None):

        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if data_format is None:
            data_format = K.image_data_format()
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        # init Y_train
        self.Y_train_data = Y_train_data
        labels = ' '.join(Y_train_data.tags).split()
        labels_u = list(set(labels))
        labels_u.sort()
        labelBinarizer = LabelBinarizer()
        labelBinarizer.fit(labels_u)
        self.labelBinarizer = labelBinarizer

        self.samples = self.Y_train_data.shape[0]
        self.num_class = len(labels_u)
        print('Found %d images belonging to %d classes.' % (self.samples, self.num_class))

        super(DirectoryGeneratorStat, self).__init__(self.samples, batch_size, shuffle, seed)

    def _tag_to_vec(self, tags):
        return self.labelBinarizer.transform(tags.split()).sum(axis=0)

    def next(self):
        index_array, current_index, current_batch_size = next(self.index_generator)

        # batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        batch_x = []
        batch_y = np.zeros((current_batch_size,) + self.labelBinarizer.classes_.shape)

        grayscale = self.color_mode == 'grayscale'
        for i, j in enumerate(index_array):
            fname = self.Y_train_data.image_name.ix[j] + '.jpg'
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            x = get_stat_features(x)
            # batch_x[i] = x
            batch_x.append(x)
            batch_y[i] = self._tag_to_vec(self.Y_train_data.tags.ix[j])

        batch_x = np.vstack(batch_x)

        return batch_x, batch_y
