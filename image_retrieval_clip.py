import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

# get list of classes in the given data set
ROOT = 'D:/03.LamTT/05.Hoc_tap/AIO2024/Module02/data'
CLASS_NAME = sorted(list(os.listdir(f'{ROOT}/train')))

# read image from folder/path, convert to RGB and resize


def read_image_from_path(path, size):
    im = Image.open(path).convert('RGB').resize(size)
    return np.array(im)


def folder_to_images(folder, size):
    list_dir = [folder + '/' + name for name in os.listdir(folder)]
    images_np = np.zeros(shape=(len(list_dir), *size, 3))
    images_path = []
    for i, path in enumerate(list_dir):
        images_np[i] = read_image_from_path(path, size)
        images_path.append(path)
    images_path = np.array(images_path)
    return images_np, images_path

# Define L1 score formula


def absolute_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.sum(np.abs(data - query), axis=axis_batch_size)


embedding_function = OpenCLIPEmbeddingFunction()


def get_single_image_embedding(image):
    embedding = embedding_function._encode_image(image=image)
    return np.array(embedding)


def get_l1_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size)
            embedding_list = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(
                    images_np[idx_img].astype(np.uint8))
                embedding_list.append(embedding)
            rates = absolute_difference(
                query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score

# Define L2 score formula


def mean_square_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.mean((data - query)**2, axis=axis_batch_size)


def get_l2_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(
                path, size)  # mang numpy nhieu anh, paths
            embedding_list = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(
                    images_np[idx_img].astype(np.uint8))
                embedding_list.append(embedding)
            rates = mean_square_difference(
                query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score

# Define cosine_similarity score


def cosine_similarity(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_norm = np.sqrt(np.sum(query**2))
    data_norm = np.sqrt(np.sum(data**2, axis=axis_batch_size))
    return np.sum(data * query, axis=axis_batch_size) / (query_norm*data_norm + np.finfo(float).eps)


def get_cosine_similarity_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(
                path, size)  # mang numpy nhieu anh, paths
            embedding_list = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(
                    images_np[idx_img].astype(np.uint8))
                embedding_list.append(embedding)
            rates = cosine_similarity(
                query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score

# correlation_coefficient score


def correlation_coefficient(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_mean = query - np.mean(query)
    data_mean = data - np.mean(data, axis=axis_batch_size, keepdims=True)
    query_norm = np.sqrt(np.sum(query_mean**2))
    data_norm = np.sqrt(np.sum(data_mean**2, axis=axis_batch_size))
    return np.sum(data_mean * query_mean, axis=axis_batch_size) / (query_norm*data_norm + np.finfo(float).eps)


def get_correlation_coefficient_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(
                path, size)  # mang numpy nhieu anh, paths
            embedding_list = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(
                    images_np[idx_img].astype(np.uint8))
                embedding_list.append(embedding)
            rates = correlation_coefficient(
                query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score

# show/plot the results


def plot_results(query_path, ls_path_score, reverse):
    sorted_ls_path_score = sorted(
        np.array(ls_path_score), key=lambda x: x[1], reverse=reverse)

    image_paths = []
    image_paths.append(query_path)
    for x in range(5):
        image_paths.append(sorted_ls_path_score[x][0])

    # Create a figure and axis array
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))  # 2 rows and 3 columns
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Loop through the images and plot them
    for ax, img_path in zip(axes, image_paths):
        img = Image.open(img_path).convert('RGB').resize((448, 448))
        ax.imshow(img)
        ax.axis('off')  # Hide axes
        # Optional: Add a title
        if image_paths.index(img_path) > 0:
            ax.set_title(f'Top {image_paths.index(img_path)}:')
        else:
            ax.set_title(f'Query Image')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


# test case with class 'Orange_easy'
root_img_path = f"{ROOT}/train/"
query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
# query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
size = (448, 448)

# test with L1 score
# query, ls_path_score = get_l1_score(root_img_path, query_path, size)
# plot_results(query_path, ls_path_score, False)

# test with L2 score
# query, ls_path_score = get_l2_score(root_img_path, query_path, size)
# plot_results(query_path, ls_path_score, False)

# test with cosine_similarity_score
query, ls_path_score = get_cosine_similarity_score(
    root_img_path, query_path, size)
plot_results(query_path, ls_path_score, True)

# test with cosine_similarity_score
# query, ls_path_score = get_correlation_coefficient_score(
#    root_img_path, query_path, size)
# plot_results(query_path, ls_path_score, True)
