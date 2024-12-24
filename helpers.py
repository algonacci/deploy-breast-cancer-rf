import pickle
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Load model and scaler
with open('best_model.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)
with open('rf_standard_model.pkl', 'rb') as standard_model_file:
    standard_model = pickle.load(standard_model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


# Define GLCM parameters
glcm_distances = [1, 2, 3, 4]
glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2]
glcm_properties = ['contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity', 'ASM']
hsv_properties = ['hue', 'saturation', 'value']

# Class names
class_names = ['ganas', 'jinak', 'normal']


def predict_image(image_path):
    # Extract features and scale them
    features = extract_features(image_path).reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Optimized model predictions
    best_probs = best_model.predict_proba(features_scaled)[0]
    best_class_index = np.argmax(best_probs)
    best_confidence_score = best_probs[best_class_index] * 100
    best_predicted_class = class_names[best_class_index]

    # Standard model predictions
    standard_probs = standard_model.predict_proba(features_scaled)[0]
    standard_class_index = np.argmax(standard_probs)
    standard_confidence_score = standard_probs[standard_class_index] * 100
    standard_predicted_class = class_names[standard_class_index]

    return {
        "optimized": {
            "predicted_class": best_predicted_class,
            "confidence_score": best_confidence_score
        },
        "unoptimized": {
            "predicted_class": standard_predicted_class,
            "confidence_score": standard_confidence_score
        }
    }


def extract_features(image_path, image_size=(256, 256)):
    # Read and process the image as described in your original code
    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv_features = []
    for property_name in hsv_properties:
        property_value = hsv_image[:, :, hsv_properties.index(property_name)].ravel()
        hsv_features.extend([np.mean(property_value), np.std(property_value)])
    glcm = graycomatrix(gray_image, distances=glcm_distances, angles=glcm_angles, symmetric=True, normed=True)
    glcm_features = []
    for property_name in glcm_properties:
        property_value = graycoprops(glcm, property_name).ravel()
        glcm_features.extend([np.mean(property_value), np.std(property_value)])
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    color_histogram_features = cv2.normalize(hist, hist).flatten()
    lbp = local_binary_pattern(gray_image, 24, 3, method="uniform")
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 24 + 3), range=(0, 24 + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)
    lbp_features = lbp_hist
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_features = [np.mean(sobelx), np.std(sobelx), np.mean(sobely), np.std(sobely)]
    combined_features = np.concatenate([hsv_features, glcm_features, color_histogram_features, lbp_features, sobel_features])
    return combined_features

def process_and_save_image(image_path):
    # Load and resize the original image
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (256, 256))
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Local Binary Pattern (LBP) - memberikan tekstur yang lebih jelas
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")
    lbp_image = (lbp / lbp.max() * 255).astype(np.uint8)
    lbp_image = cv2.merge([lbp_image]*3)
    
    # Sobel Edge Detection - untuk deteksi tepi
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobelx, sobely)
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sobel_edges = cv2.merge([sobel_edges]*3)
    
    # Adaptive Thresholding - membantu memisahkan area penting
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    adaptive_thresh = cv2.merge([adaptive_thresh]*3)
    
    # Original grayscale image
    gray_3ch = cv2.merge([gray_image]*3)
    
    # Combine images horizontally (2x2 grid)
    top_row = np.hstack([gray_3ch, sobel_edges])
    bottom_row = np.hstack([adaptive_thresh, lbp_image])
    
    # Combine rows vertically
    combined_image = np.vstack([top_row, bottom_row])
    
    return combined_image

