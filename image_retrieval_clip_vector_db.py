import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
import chromadb
from tqdm import tqdm

# get list of classes in the given data set
ROOT = 'D:/03.LamTT/05.Hoc_tap/AIO2024/Module02/data'
CLASS_NAME = sorted(list(os.listdir(f'{ROOT}/train')))
HNSW_SPACE = "hnsw:space"

# get file paths of all images in train data set


def get_files_path(path):
    files_path = []
    for label in CLASS_NAME:
        label_path = path + "/" + label
        filenames = os.listdir(label_path)
        for filename in filenames:
            filepath = label_path + '/' + filename
            files_path.append(filepath)
    return files_path


data_path = f'{ROOT}/train'
files_path = get_files_path(path=data_path)

# extract and store features of all images into a vector database
embedding_function = OpenCLIPEmbeddingFunction()


def get_single_image_embedding(image):
    embedding = embedding_function._encode_image(image=np.array(image))
    return embedding


def add_embedding(collection, files_path):
    ids = []
    embeddings = []
    for id_filepath, filepath in tqdm(enumerate(files_path)):
        ids.append(f'id_{id_filepath}')
        image = Image.open(filepath)
        embedding = get_single_image_embedding(image=image)
        embeddings.append(embedding)
    collection.add(embeddings=embeddings, ids=ids)


# Create a Chroma Client
chroma_client = chromadb.Client()

# Create a collection
l2_collection = chroma_client.get_or_create_collection(
    name="l2_collection", metadata={HNSW_SPACE: "l2"})
add_embedding(collection=l2_collection, files_path=files_path)

# Create a collection
cosine_collection = chroma_client.get_or_create_collection(
    name="Cosine_collection", metadata={HNSW_SPACE: "cosine"})
add_embedding(collection=cosine_collection, files_path=files_path)


def search(image_path, collection, n_results):
    query_image = Image.open(image_path)
    query_embedding = get_single_image_embedding(query_image)
    results = collection.query(query_embeddings=[query_embedding],
                               n_results=n_results)  # how many results to return
    return results

# show/plot the results


def plot_results(image_path, files_path, results):
    query_image = Image.open(image_path).resize((448, 448))
    images = [query_image]
    class_name = []
    for id_img in results['ids'][0]:
        id_img = int(id_img.split('_')[-1])
        img_path = files_path[id_img]
        img = Image.open(img_path).resize((448, 448))
        images.append(img)
        class_name.append(img_path.split('/')[2])

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Iterate through images and plot them
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        if i == 0:
            ax.set_title(f"Query Image: {image_path.split('/')[2]}")
        else:
            ax.set_title(f"Top {i+1}: {class_name[i-1]}")
        ax.axis('off')  # Hide axes
    # Display the plot
    plt.show()


# test
test_path = f'{ROOT}/test'
test_files_path = get_files_path(path=test_path)
test_path = test_files_path[1]
# test with l2 collection
# l2_results = search(image_path=test_path,
#                    collection=l2_collection, n_results=5)
# plot_results(image_path=test_path, files_path=files_path, results=l2_results)
# test with cosine collection
l2_results = search(image_path=test_path,
                    collection=cosine_collection, n_results=5)
plot_results(image_path=test_path, files_path=files_path, results=l2_results)
