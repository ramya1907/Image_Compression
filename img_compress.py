import numpy as np
import matplotlib.pyplot as plt
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
combo = str


def ch_score():
    original_image = PIL.Image.open(image_path)
    resized_image = original_image.resize((300, 300))
    image_array = np.array(resized_image)
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
    resized_image = original_image.resize((300, 300))
    image_array = np.array(resized_image)
    flattened_image_array = image_array.reshape(-1, 3)
    silhouette = []
    for k in range(2, 10):
        km = KMeans(n_clusters=k, n_init=25, random_state=1234)
        km.fit(flattened_image_array)
        silhouette.append(silhouette_score(flattened_image_array, km.labels_))

    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 10), silhouette)
    plt.scatter(range(2, 10), silhouette, s=150)
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


def save_compressed_image(compressed_photo, k_input):
    save_path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
    if not save_path:
        return

    if save_path:
        compressed_photo.save(save_path)
        print(f"Compressed image (k = {k_input}) saved successfully.")


def compress_image(k_input, original_image):
    if not isinstance(k_input, int):
        k_input = 16

    if original_image is None:
        original_image = PIL.Image.open(image_path)

    original_image = original_image.resize((1000, 1000))
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

    compressed_photo = PIL.ImageTk.PhotoImage(compressed_image.resize((300, 300)))

    comp_image_label = ttk.Label(compress_frame, image=compressed_photo)
    comp_image_label.grid(column=0, row=1, padx=10, pady=10)

    comp_image_label.image = compressed_photo

    save_btn = ttk.Button(frm, text="Save", command=lambda: save_compressed_image(compressed_image, k_input))
    save_btn.grid(column=1, row=5, padx=10, pady=10)

    plt.show()


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


def reset():
    entry.delete(0, END)
    combo.current(0)

def on_combobox_select(event):
    global combo
    global option
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


root = Tk()
root.geometry("1000x800")

frm = ttk.Frame(root, padding=10)
frm.grid()

ttk.Label(frm, text="Image Compression").grid(column=0, row=0, padx=(80, 0), pady=10)

text_widget = Text(frm, height=8, width=80)
text_widget.grid(column=0, row=1, padx=(80, 0), pady=10)
text_widget.insert(END, "This application uses the machine learning algorithm: k-means clustering \n")
text_widget.insert(END, "\nto perform image compression. \n")
text_widget.insert(END, "\nPython libraries used include: Numpy, Matplotlib Pyplot, \n")
text_widget.insert(END, "\nSci-kit Learn and Python Imaging Library(PIL) \n")

text_widget.config(state=DISABLED)

frm3 = ttk.Frame(frm, padding=10)
frm3.grid(column=1, row=1)
ttk.Button(frm3, text="Reset", command=reset).grid(column=0, row=0, padx=10, pady=10)
ttk.Button(frm3, text="Quit", command=root.destroy).grid(column=1, row=0, padx=10, pady=10)

frm0 = ttk.Frame(frm, padding=10)
frm0.grid(column=0, row=2)
ttk.Label(frm0, text="Use your own image").grid(column=0, row=0, padx=10, pady=10)
open_btn = ttk.Button(frm0, text="Open Image", command=lambda: open_image(image_label))
open_btn.grid(column=1, row=0, padx=10, pady=10)

image_label = Label(frm)
image_label.grid(column=0, row=3, padx=10, pady=10)

temp_image = PIL.Image.open(image_path)
temp_image = temp_image.resize((300, 300))
temp_photo = PIL.ImageTk.PhotoImage(temp_image)
image_label.configure(image=temp_photo)
image_label.image = temp_photo

frm1 = ttk.Frame(frm, padding=10)
frm1.grid(column=1, row=2)
ttk.Label(frm1, text="Insert value for k:").grid(column=0, row=0, sticky='e', padx=10, pady=10)
entry = Entry(frm1, width=10)
entry.grid(column=1, row=0, padx=10, pady=10)
entry.focus_set()

compress_frame = ttk.Frame(frm, padding=5)
compress_frame.grid(column=1, row=3)
compress_btn = ttk.Button(compress_frame, text="Display Compressed Image",
                          command=lambda: compress_image(entry.get(), og_image))
compress_btn.grid(column=0, row=0, padx=10, pady=10)

combo_frame = ttk.Frame(frm, padding=10)
combo_frame.grid(column=0, row=7)
combo_label = ttk.Label(combo_frame, text="Find optimum value of k:")
combo_label.grid(column=0, row=0, padx=10, pady=10)
combo = ttk.Combobox(combo_frame, values=["Calculate WCSS", "Avg Silhouette Score", "Calinski Harabasz"],
                     state="readonly")
combo.grid(column=1, row=0, padx=10, pady=10)
combo.current(0)  # Set the default selected option

combo.bind("<<ComboboxSelected>>", on_combobox_select)

root.mainloop()
