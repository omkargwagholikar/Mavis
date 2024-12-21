import numpy as np
from PIL import Image
import cv2

def load_image(path, grayscale=True):
    """Load image and convert to grayscale numpy array if specified."""
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    return np.array(img)

def save_image(img_array, path):
    """Save numpy array as image."""
    Image.fromarray(np.uint8(img_array)).save(path)

def embed_watermark(original_img, watermark_img, alpha=0.1):
    """
    Embed watermark into original image using SVD.

    Parameters:
    original_img: numpy array of original image
    watermark_img: numpy array of watermark image (must be same size as original)
    alpha: strength of watermark (0-1)

    Returns:
    watermarked_img: numpy array of watermarked image
    U1, S1, VT1: SVD components of original image (needed for extraction)
    """
    # Ensure images are same size
    assert original_img.shape == watermark_img.shape, "Images must be same size"

    # Apply SVD to original image
    U1, S1, VT1 = np.linalg.svd(original_img, full_matrices=False)

    # Apply SVD to watermark
    U2, S2, VT2 = np.linalg.svd(watermark_img, full_matrices=False)

    # Modify singular values of original image
    S_modified = S1 + (alpha * S2)

    # Reconstruct watermarked image
    watermarked_img = np.dot(U1 * S_modified, VT1)

    # Clip values to valid range
    watermarked_img = np.clip(watermarked_img, 0, 255)

    return watermarked_img, U1, S1, VT1

def extract_watermark(watermarked_img, U1, S1, VT1, alpha=0.1):
    """
    Extract watermark from watermarked image using original SVD components.

    Parameters:
    watermarked_img: numpy array of watermarked image
    U1, S1, VT1: SVD components of original image
    alpha: strength used during embedding

    Returns:
    extracted_watermark: numpy array of extracted watermark
    """
    # Apply SVD to watermarked image
    U2, S2, VT2 = np.linalg.svd(watermarked_img, full_matrices=False)

    # Extract watermark singular values
    S_extracted = (S2 - S1) / alpha

    # Reconstruct watermark using original U and VT
    extracted_watermark = np.dot(U2 * S_extracted, VT2)

    # Normalize to 0-255 range
    extracted_watermark = ((extracted_watermark - extracted_watermark.min()) * 255
                         / (extracted_watermark.max() - extracted_watermark.min()))

    return np.clip(extracted_watermark, 0, 255)

def main():
    # Example usage
    # Load images
    original = load_image('../base.jpeg', grayscale=True)
    watermark = load_image('../watermark.jpeg', grayscale=True)

    # Resize watermark to match original if needed
    if original.shape != watermark.shape:
        watermark = cv2.resize(watermark, (original.shape[1], original.shape[0]))

    # Embed watermark
    watermarked_img, U1, S1, VT1 = embed_watermark(original, watermark, alpha=0.1)

    # Save watermarked image
    save_image(watermarked_img, 'watermarked.png')

    # Extract watermark
    extracted_watermark = extract_watermark(watermarked_img, U1, S1, VT1, alpha=0.1)

    # Save extracted watermark
    save_image(extracted_watermark, 'extracted_watermark.png')

if __name__ == "__main__":
    main()
