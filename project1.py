import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# =========================
# LOAD IMAGE
# =========================
Tk().withdraw()
image_path = askopenfilename(
    title="Pilih Gambar",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if not image_path:
    print("Tidak ada file dipilih!")
    exit()

original = cv2.imread(image_path)

if original is None:
    print("Gagal membaca gambar!")
    exit()

# Resize otomatis jika terlalu besar
max_width = 700
h, w = original.shape[:2]
if w > max_width:
    scale = max_width / w
    original = cv2.resize(original, (int(w*scale), int(h*scale)))

# =========================
# FILTER FUNCTIONS
# =========================

def grayscale(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, 1-intensity, gray, intensity, 0)

def sepia(img, intensity):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sep = cv2.transform(img, kernel)
    sep = np.clip(sep, 0, 255)
    return cv2.addWeighted(img, 1-intensity, sep.astype(np.uint8), intensity, 0)

def negative(img, intensity):
    neg = 255 - img
    return cv2.addWeighted(img, 1-intensity, neg, intensity, 0)

def brightness(img, intensity):
    return cv2.convertScaleAbs(img, alpha=1, beta=intensity*100)

def vignette(img, intensity):
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols*intensity+1)
    kernel_y = cv2.getGaussianKernel(rows, rows*intensity+1)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    vignette_img = np.copy(img)
    for i in range(3):
        vignette_img[:,:,i] = vignette_img[:,:,i] * mask
    return vignette_img.astype(np.uint8)

def blur_artistic(img, intensity):
    k = int(1 + intensity*25)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)

# =========================
# HISTOGRAM RGB
# =========================

plt.ion()
fig, ax = plt.subplots()

def show_rgb_histogram(img):
    ax.clear()
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [256], [0,256])
        ax.plot(hist)
    ax.set_title("Histogram RGB")
    ax.set_xlim([0,256])
    plt.draw()
    plt.pause(0.001)

# =========================
# GUI WINDOW
# =========================

cv2.namedWindow("Roushan Filter App")
cv2.createTrackbar("Intensity", "Roushan Filter App", 50, 100, lambda x: None)
cv2.createTrackbar("Filter (0-5)", "Roushan Filter App", 0, 5, lambda x: None)

print("0 = Grayscale")
print("1 = Sepia")
print("2 = Negative")
print("3 = Brightness")
print("4 = Vignette")
print("5 = Blur Artistik")
print("Tekan S untuk simpan")
print("Tekan ESC untuk keluar")

# =========================
# MAIN LOOP
# =========================

while True:
    intensity = cv2.getTrackbarPos("Intensity", "Roushan Filter App") / 100
    filter_choice = cv2.getTrackbarPos("Filter (0-5)", "Roushan Filter App")

    if filter_choice == 0:
        result = grayscale(original, intensity)
    elif filter_choice == 1:
        result = sepia(original, intensity)
    elif filter_choice == 2:
        result = negative(original, intensity)
    elif filter_choice == 3:
        result = brightness(original, intensity)
    elif filter_choice == 4:
        result = vignette(original, max(intensity, 0.1))
    elif filter_choice == 5:
        result = blur_artistic(original, intensity)

    # Tambah Border
    bordered = cv2.copyMakeBorder(
        result, 20, 20, 20, 20,
        cv2.BORDER_CONSTANT,
        value=[255,255,255]
    )

    # Watermark Elegan
    text = "Roushan Filter App"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    x = bordered.shape[1] - text_width - 20
    y = bordered.shape[0] - 20

    # Shadow
    cv2.putText(
        bordered,
        text,
        (x+2, y+2),
        font,
        font_scale,
        (0,0,0),
        thickness+1,
        cv2.LINE_AA
    )

    # Main text
    cv2.putText(
        bordered,
        text,
        (x, y),
        font,
        font_scale,
        (255,255,255),
        thickness,
        cv2.LINE_AA
    )

    cv2.imshow("Roushan Filter App", bordered)

    show_rgb_histogram(result)

    key = cv2.waitKey(1)

    if key == 27:
        break

    if key == ord('s'):
        cv2.imwrite(
            "hasil_final.jpg",
            bordered,
            [cv2.IMWRITE_JPEG_QUALITY, 100]
        )
        print("Disimpan sebagai hasil_final.jpg")

cv2.destroyAllWindows()
plt.close()