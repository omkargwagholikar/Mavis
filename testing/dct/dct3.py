# Base image watermarking and extraction seems to be working
# need to separate the functions from each other

import numpy as np
import pywt
from PIL import Image
from scipy.fftpack import dct, idct
from pathlib import Path

class WaveletDCTWatermark:
    def __init__(self, base_path=None):
        """Initialize the watermarking system with base path"""
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.dataset_path = self.base_path / 'dataset'
        self.result_path = self.base_path / 'result'

        # Create necessary directories
        self.dataset_path.mkdir(exist_ok=True)
        self.result_path.mkdir(exist_ok=True)

    def convert_image(self, image_path, size, to_grayscale=False):
        """Convert and resize image, with option to convert to grayscale"""
        try:
            img = Image.open(image_path).resize((size, size), Image.Resampling.LANCZOS)
            if to_grayscale:
                img = img.convert('L')
                image_array = np.array(img.getdata(), dtype=np.float64).reshape((size, size))
            else:
                image_array = np.array(img, dtype=np.float64)

            # Save processed image
            processed_path = self.dataset_path / Path(image_path).name
            img.save(processed_path)

            return image_array
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            raise

    def process_coefficients(self, image_array, model, level):
        """Process wavelet coefficients for each color channel if RGB"""
        try:
            if len(image_array.shape) == 3:  # RGB image
                coeffs_by_channel = []
                for channel in range(3):
                    coeffs = pywt.wavedec2(data=image_array[:,:,channel], wavelet=model, level=level)
                    coeffs_by_channel.append(list(coeffs))
                return coeffs_by_channel
            else:  # Grayscale image
                coeffs = pywt.wavedec2(data=image_array, wavelet=model, level=level)
                return list(coeffs)
        except Exception as e:
            print(f"Error processing coefficients: {str(e)}")
            raise

    def embed_watermark(self, watermark_array, orig_image):
        """Embed watermark in DCT coefficients"""
        try:
            watermark_flat = watermark_array.ravel()
            ind = 0

            for x in range(0, orig_image.shape[0], 8):
                for y in range(0, orig_image.shape[1], 8):
                    if ind < len(watermark_flat):
                        subdct = orig_image[x:x+8, y:y+8].copy()
                        subdct[5][5] = watermark_flat[ind]
                        orig_image[x:x+8, y:y+8] = subdct
                        ind += 1

            return orig_image
        except Exception as e:
            print(f"Error embedding watermark: {str(e)}")
            raise

    def apply_dct(self, image_array):
        """Apply DCT transform to image"""
        try:
            size = image_array.shape[0]
            all_subdct = np.empty((size, size), dtype=np.float64)

            for i in range(0, size, 8):
                for j in range(0, size, 8):
                    subpixels = image_array[i:i+8, j:j+8]
                    subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
                    all_subdct[i:i+8, j:j+8] = subdct

            return all_subdct
        except Exception as e:
            print(f"Error applying DCT: {str(e)}")
            raise

    def inverse_dct(self, all_subdct):
        """Apply inverse DCT transform"""
        try:
            size = all_subdct.shape[0]
            all_subidct = np.empty((size, size), dtype=np.float64)

            for i in range(0, size, 8):
                for j in range(0, size, 8):
                    subidct = idct(idct(all_subdct[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
                    all_subidct[i:i+8, j:j+8] = subidct

            return all_subidct
        except Exception as e:
            print(f"Error applying inverse DCT: {str(e)}")
            raise

    def get_watermark(self, dct_watermarked_coeff, watermark_size):
        """Extract watermark from DCT coefficients"""
        try:
            subwatermarks = []

            for x in range(0, dct_watermarked_coeff.shape[0], 8):
                for y in range(0, dct_watermarked_coeff.shape[1], 8):
                    coeff_slice = dct_watermarked_coeff[x:x+8, y:y+8]
                    subwatermarks.append(coeff_slice[5][5])

            watermark = np.array(subwatermarks).reshape(watermark_size, watermark_size)
            return watermark
        except Exception as e:
            print(f"Error extracting watermark: {str(e)}")
            raise

    def recover_watermark(self, image_array, model='haar', level=1):
        """Recover watermark from watermarked image"""
        try:
            coeffs_watermarked_image = self.process_coefficients(image_array, model, level)

            if isinstance(coeffs_watermarked_image[0], list):  # RGB image
                # Use green channel for watermark recovery
                dct_watermarked_coeff = self.apply_dct(coeffs_watermarked_image[1][0])
            else:
                dct_watermarked_coeff = self.apply_dct(coeffs_watermarked_image[0])

            watermark_array = self.get_watermark(dct_watermarked_coeff, 128)
            watermark_array = np.uint8(watermark_array)

            # Save recovered watermark
            img = Image.fromarray(watermark_array)
            img.save(self.result_path / 'recovered_watermark.jpg')
        except Exception as e:
            print(f"Error recovering watermark: {str(e)}")
            raise

    def save_image(self, image_array, name):
        """Save image array as image file"""
        try:
            image_array_copy = image_array.clip(0, 255)
            image_array_copy = image_array_copy.astype("uint8")
            img = Image.fromarray(image_array_copy)
            img.save(self.result_path / name)
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            raise

    def watermark_image(self, image_path, watermark_path):
        """Main watermarking process"""
        try:
            model = 'haar'
            level = 1

            print("Converting images...")
            image_array = self.convert_image(image_path, 2048, to_grayscale=False)
            watermark_array = self.convert_image(watermark_path, 128, to_grayscale=True)

            print("Processing and embedding watermark...")
            coeffs_image = self.process_coefficients(image_array, model, level)

            # Handle each color channel separately
            watermarked_image = np.empty_like(image_array)
            for channel in range(3):
                dct_array = self.apply_dct(coeffs_image[channel][0])
                # Only embed watermark in green channel (index 1)
                if channel == 1:
                    dct_array = self.embed_watermark(watermark_array, dct_array)
                coeffs_image[channel][0] = self.inverse_dct(dct_array)
                # Reconstruct each channel
                watermarked_image[:,:,channel] = pywt.waverec2(coeffs_image[channel], model)

            print("Saving watermarked image...")
            self.save_image(watermarked_image, 'image_with_watermark.jpg')

            return watermarked_image
        except Exception as e:
            print(f"Error in watermarking process: {str(e)}")
            raise

def main():
    """Example usage"""
    try:
        # Initialize watermarking system
        watermarker = WaveletDCTWatermark()

        # Get input paths
        image_path = Path("../base.jpeg")
        watermark_path = Path("../watermark.jpeg")

        # Validate paths
        if not image_path.exists():
            raise FileNotFoundError(f"Original image not found: {image_path}")
        if not watermark_path.exists():
            raise FileNotFoundError(f"Watermark image not found: {watermark_path}")

        # Process watermarking
        print("\nProcessing watermark...")
        watermarked_image = watermarker.watermark_image(image_path, watermark_path)

        print("Extracting watermark...")
        watermarker.recover_watermark(image_array=watermarked_image)

        print("\nResults saved:")
        print("- Watermarked image: ./result/image_with_watermark.jpg")
        print("- Recovered watermark: ./result/recovered_watermark.jpg")

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Watermarking process failed.")

if __name__ == "__main__":
    main()
