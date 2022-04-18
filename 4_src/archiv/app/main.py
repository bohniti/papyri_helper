import backend
import streamlit as st
import numpy as np
import helpers

st.set_page_config(layout="wide")
samples = '/Users/beantown/PycharmProjects/master-thesis/data/preprocessed/cleansed'
labels = '/Users/beantown/PycharmProjects/master-thesis/data/preprocessed/cleansed.csv'

dataset = backend.load_dataset(samples, labels)
labels_to_indices, labels = backend.load_indices_and_labels(dataset)

st.title("Papyri Finder")

papyri_id = st.sidebar.selectbox('Papyri ID:', labels)
st.write('You selected papyri:', papyri_id)

query_class = labels_to_indices[papyri_id]
samples_count = len(query_class)
sample_numbers = np.arange(samples_count)

if query_class is not None:
    query_imgs = []
    visualize_imgs = []
    for i in sample_numbers:
        query_img, visualize_img = backend.get_samples_images(dataset, query_class, i)
        query_imgs.append(query_img)
        visualize_imgs.append(visualize_img)

sample_iterator = helpers.paginator(label=f'Papyri samples of Papyri {papyri_id}',
                                    items=visualize_imgs,
                                    items_per_page=3,
                                    on_sidebar=True)

indices_on_page, images_on_page = map(list, zip(*sample_iterator))
st.image(images_on_page, width=600, caption=indices_on_page)
query_sample = st.sidebar.selectbox("Select a query sample.", sample_numbers)

k = st.sidebar.selectbox('Get the k nearest papyri fragments:', np.arange(10, 100))
if query_sample is not None:

    if query_img is not None:
        if k is not None:
            print(k)
            predictions = backend.get_predictions(dataset, query_img, int(k))

            sample_iterator = helpers.paginator(label=f'{k}-Nearest Fragments of sample #{query_sample} from Papyri {papyri_id}',
                                            items=predictions,
                                            items_per_page=100,
                                            on_sidebar=False)

            indices_on_page, images_on_page = map(list, zip(*sample_iterator))
            st.image(images_on_page, width=600, caption=indices_on_page)







