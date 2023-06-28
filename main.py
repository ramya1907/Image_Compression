import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

import PIL.Image
import PIL.ImageTk

from tkinter import *
from tkinter import ttk
from tkinter import filedialog

image_path = './cat.jpg'
og_image = None
combo = None


def ch_score():
    original_image = PIL.Image.open(image_path)
    image_array = np.array(original_image)
    flattened_image_array = image_array.reshape(-1, 3)

    calinski = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=25, random_state=1234)
        km.fit(flattened_image_array)
        calinski.append(calinski_harabasz_score(flattened_image_array, km.labels_))

    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 11), calinski)
    plt.scatter(range(2, 11), calinski, s=150)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Calinski Harabasz Score')
    plt.title('Calinski Harabasz Score vs k')
    plt.show()


def avg_sil_score():
    original_image = PIL.Image.open(image_path)
    image_array = np.array(original_image)
    flattened_image_array = image_array.reshape(-1, 3)
    silhouette = []
    for k in range(2, 16):
        km = KMeans(n_clusters=k, n_init=25, random_state=1234)
        km.fit(flattened_image_array)
        silhouette.append(silhouette_score(flattened_image_array, km.labels_))

    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 16), silhouette)
    plt.scatter(range(2, 16), silhouette, s=150)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.title('Average Silhouette Score vs k')
    plt.show()


def calculate_wcss():
    original_image = PIL.Image.open(image_path)
    image_array = np.array(original_image)
    flattened_image_array = image_array.reshape(-1, 3)  # Assuming RGB image

    # Calculate the WCSS for each value of k
    wcss = []
    for k in range(2, 16):
        kmeans = KMeans(n_clusters=k, n_init=25, random_state=42).fit(flattened_image_array)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 16), wcss)
    plt.scatter(range(2, 16), wcss, s=150)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within Cluster Sum of Squares (WCSS)')
    plt.title('WCSS vs Number of Clusters')
    plt.show()


def compress_image(k_input, original_image):

    if not isinstance(k_input, int):
        k_input = 16

    if original_image is None:
        original_image = PIL.Image.open(image_path)

    plt.figure(figsize=(8, 6))
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()

    # Convert the image to a numpy array
    image_array = np.array(original_image)
    #  Flatten the image array
    flattened_image_array = image_array.reshape(-1, 3)  # Assuming RGB image

    # Create and fit the k-means clustering model
    kmeans = KMeans(n_clusters=k_input, n_init=25, random_state=42).fit(flattened_image_array)

    # Assign cluster labels to each pixel
    cluster_labels = kmeans.labels_
    # Get the cluster centers (representative colors)
    cluster_centers = kmeans.cluster_centers_

    # Replace each pixel value with the corresponding cluster center
    compressed_image_array = cluster_centers[cluster_labels]
    # Reshape the quantized image array to the original shape
    compressed_image_array = compressed_image_array.reshape(image_array.shape)
    # Convert the array back to an image object
    compressed_image = PIL.Image.fromarray(np.uint8(compressed_image_array))

    plt.figure(figsize=(8, 6))
    plt.imshow(compressed_image)
    plt.title('Compressed Image (k = {})'.format(k_input))
    plt.axis('off')

    # Add save button
    save_button_ax = plt.axes([0.85, 0.05, 0.1, 0.05])
    save_button = Button(save_button_ax, 'Save Image')
    save_button.on_clicked(save_compressed_image(compressed_image, k_input))

    plt.show()


def save_compressed_image(image, k_input):
    save_path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
    if save_path:
        image.save(save_path)
        print(f"Compressed image (k = {k_input}) saved successfully.")


def open_image(image_label):
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        image = PIL.Image.open(file_path)
        # Resize the image to fit the display area
        image = image.resize((300, 300))
        global og_image
        og_image = image
        photo = PIL.ImageTk.PhotoImage(image)
        image_label.configure(image=photo)
        image_label.image = photo


def reset(entry):
    entry.delete(0, END)


def on_combobox_select(event):
    global combo
    selected_option = combo.get()
    if selected_option == "Calculate WCSS":
        print("Option 1 selected")
        calculate_wcss()

    elif selected_option == "Avg Silhouette Score":
        print("Option 2 selected")
        avg_sil_score()

    elif selected_option == "Calinski Harabasz":
        print("Option 3 selected")
        ch_score()


def display():
    root = Tk()
    frm = ttk.Frame(root, padding=10)
    frm.grid()

    ttk.Label(frm, text="Image Compression").grid(column=0, row=0)

    text_widget = Text(frm, height=5, width=80)
    text_widget.grid(column=0, row=1, padx=10, pady=10)
    text_widget.insert(END, "This application was created to demonstrate skills in utilizing\n")
    text_widget.insert(END, "k-means to perform image compression as well as knowledge of\n")
    text_widget.insert(END, "matplotlib and numpy.\n")
    text_widget.config(state=DISABLED)

    frm0 = ttk.Frame(frm, padding=10)
    frm0.grid(column=0, row=2)
    ttk.Label(frm0, text="Use your own image").grid(column=0, row=0, padx=10, pady=10)
    open_btn = ttk.Button(frm0, text="Open Image", command=lambda: open_image(image_label))
    open_btn.grid(column=1, row=0, padx=10, pady=10)

    image_label = Label(frm)
    image_label.grid(column=0, row=3, padx=10, pady=10)

    frm1 = ttk.Frame(frm, padding=10)
    frm1.grid(column=0, row=4)
    ttk.Label(frm1, text="Insert value for k:").grid(column=0, row=0, sticky='e', padx=10, pady=10)
    entry = Entry(frm1, width=10)
    entry.grid(column=1, row=0, padx=10, pady=10)
    entry.focus_set()

    global og_image
    compress_btn = ttk.Button(frm, text="Display Compressed Image",
                              command=lambda: compress_image(entry.get(), og_image))
    compress_btn.grid(column=0, row=5, padx=10, pady=10)

    global combo
    combo = ttk.Combobox(frm, values=["Calculate WCSS", "Avg Silhouette Score", "Calinski Harabasz"], state="readonly")
    combo.grid(column=0, row=6, padx=10, pady=10)
    combo.current(0)  # Set the default selected option

    combo.bind("<<ComboboxSelected>>", on_combobox_select)

    frm3 = ttk.Frame(frm, padding=10)
    frm3.grid(column=0, row=7)
    ttk.Button(frm3, text="Reset", command=lambda: reset(entry)).grid(column=0, row=0, padx=10, pady=10)
    ttk.Button(frm3, text="Quit", command=root.destroy).grid(column=1, row=0, padx=10, pady=10)

    root.mainloop()


def main():
    return


display()
