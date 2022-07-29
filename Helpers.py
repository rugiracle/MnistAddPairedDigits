import torch
import matplotlib.pyplot as plt
import random
import numpy as np
RANDOM_SEED = 50
N_CLASSES = 19


def plot_dataset_new_classes(data, name):
    names = list(data.keys())
    values = list(data.values())
    plt.xticks(rotation=90)
    plt.rcParams.update({'font.size': 11})
    plt.bar(range(len(data)), values, tick_label=names)
    naming = 'update/' + name + '_data_distribution.png'
    # plt.savefig(naming)
    # plt.clf()
    plt.show()


class dataGeneration():
    def pre_process_imgs(img1, img2):
        """ resizes images and Concatenate them(for display)"""
        return torch.cat((img1, img2), -1)

    def plot_mnist_dist(mnist_dataset):
        labels_mnist = {}  # mnist distribution
        for i in range(10):
            labels_mnist[i] = 0
        for i in range(len(mnist_dataset)):
            img, label = mnist_dataset[i]
            labels_mnist[label] += 1
        plot_dataset_new_classes(labels_mnist, 'mnist')

    def plot_sample_data(data):
        ROW_IMG = 10
        N_ROWS = 5
        ids = random.sample(range(len(data)), ROW_IMG * N_ROWS + 1)
        fig = plt.figure()
        for index in range(1, ROW_IMG * N_ROWS + 1):
            plt.subplot(N_ROWS, ROW_IMG, index)
            plt.axis('off')
            img, label = data[ids[index]]
            plt.title(str(label), color='b')
            plt.imshow(img[0])
        fig.suptitle('Newly formed Dataset - preview')
        plt.show()
        # plt.savefig('update/data_sample_train_31.png')
        # plt.clf()

    def pre_process_dataset_img_channel_Random(in_dataset):
        """ Randomly pairs two digits and labels them by the sum(sum of their labels """
        torch.manual_seed(RANDOM_SEED)
        indices = random.sample(range(len(in_dataset)), len(in_dataset))
        dataset = []
        keys = range(N_CLASSES)
        labels = {}  # new labels
        extended_labels = {}  # true distribution of  the newly generated samples

        for k in keys:
            labels[k] = 0

        for i in range(10):  # 10 digits
            for j in range(10):
                extended_labels[str(i) + str(j)] = 0

        for i in range(1, len(indices) - 1, 2):
            sample_1 = i - 1
            sample_2 = i
            sample_3 = i + 1

            image_1, label_1 = in_dataset[indices[sample_1]]
            image0, label0 = in_dataset[indices[sample_2]]
            image1, label1 = in_dataset[indices[sample_3]]

            label_10 = label_1 + label0
            label01 = label0 + label1
            label_11 = label_1 + label1

            dataset.append((image_1, image0, label_10))
            dataset.append((image0, image1, label01))
            dataset.append((image_1, image1, label_11))
            """ for display : samples per combination"""
            labels[label_10] += 1
            labels[label01] += 1
            labels[label_11] += 1

            extended_labels[str(label_1) + str(label0)] += 1
            extended_labels[str(label0) + str(label1)] += 1
            extended_labels[str(label_1) + str(label1)] += 1

        # print('labels \n', labels)
        plot_dataset_new_classes(extended_labels, 'extended')
        plot_dataset_new_classes(labels, 'labels')
        return labels, dataset

    def get_weights_inverse_num_of_samples(no_of_classes, samples_per_cls, power=1):
        weights_for_samples = 1.0 / np.array(np.power(samples_per_cls, power))
        weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * no_of_classes
        # print('weights per class', weights_for_samples)
        return weights_for_samples