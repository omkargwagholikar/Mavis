import numpy as np
import cv2
from scipy.fft import dct, idct
from PIL import Image
import random
import matplotlib.pyplot as plt

class DCTWatermark:
    def __init__(self, block_size=8, alpha=15):
        """Initialize DCT watermarking system"""
        self.block_size = block_size
        self.alpha = alpha
        self.seed = None

    def _apply_dct_to_block(self, block):
        """Apply 2D DCT to a block."""
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def _apply_idct_to_block(self, block):
        """Apply 2D inverse DCT to a block."""
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    def _get_mid_band_coords(self):
        """Get coordinates of mid-frequency DCT coefficients."""
        coords = []
        for i in range(1, self.block_size-1):
            for j in range(1, self.block_size-1):
                if i + j >= 4 and i + j <= 8:
                    coords.append((i, j))
        return coords

    def process_watermark(self, watermark_path, target_size):
        """
        Process watermark image to binary sequence

        Parameters:
        watermark_path: Path to watermark image
        target_size: Required size of binary sequence

        Returns:
        binary_watermark: Binary sequence
        """
        # Read and convert watermark to binary
        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        watermark = cv2.resize(watermark, (64, 64))  # Resize for consistent processing
        _, binary_watermark = cv2.threshold(watermark, 127, 1, cv2.THRESH_BINARY)

        # Repeat or truncate to match target size
        binary_sequence = binary_watermark.flatten()
        if len(binary_sequence) < target_size:
            binary_sequence = np.tile(binary_sequence, target_size // len(binary_sequence) + 1)
        return binary_sequence[:target_size]

    def embed(self, image_path, watermark_path=None):
        """
        Embed watermark into color image

        Parameters:
        image_path: Path to input image
        watermark_path: Path to watermark image (optional)

        Returns:
        watermarked_image: Watermarked color image
        """
        # Read color image
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Pad image if necessary
        if height % self.block_size != 0 or width % self.block_size != 0:
            height_pad = self.block_size - (height % self.block_size)
            width_pad = self.block_size - (width % self.block_size)
            image = np.pad(image, ((0, height_pad), (0, width_pad), (0, 0)), mode='edge')
            height, width = image.shape[:2]

        # Calculate watermark size needed
        blocks_h = height // self.block_size
        blocks_w = width // self.block_size
        mid_band_coords = self._get_mid_band_coords()
        watermark_length = blocks_h * blocks_w * len(mid_band_coords)

        # Get watermark sequence
        if watermark_path:
            watermark = self.process_watermark(watermark_path, watermark_length)
            self.seed = None  # Not using random seed when watermark is provided
        else:
            self.seed = random.randint(0, 1000000)
            watermark = np.random.choice([0, 1], size=watermark_length)

        # Process each color channel
        watermarked_image = np.zeros_like(image, dtype=float)
        for channel in range(3):
            img_channel = image[:, :, channel].astype(float)
            watermark_idx = 0

            for i in range(0, height, self.block_size):
                for j in range(0, width, self.block_size):
                    block = img_channel[i:i+self.block_size, j:j+self.block_size]
                    dct_block = self._apply_dct_to_block(block)

                    for coord in mid_band_coords:
                        if watermark_idx < len(watermark):
                            if watermark[watermark_idx] == 1:
                                dct_block[coord] += self.alpha
                            else:
                                dct_block[coord] -= self.alpha
                            watermark_idx += 1

                    watermarked_block = self._apply_idct_to_block(dct_block)
                    watermarked_image[i:i+self.block_size, j:j+self.block_size, channel] = watermarked_block

        # Clip values and convert to uint8
        watermarked_image = np.clip(watermarked_image, 0, 255)
        return watermarked_image.astype(np.uint8)

    def extract(self, watermarked_image_path, output_path='extracted_watermark.png'):
        """
        Extract and visualize watermark from image

        Parameters:
        watermarked_image_path: Path to watermarked image
        output_path: Path to save extracted watermark

        Returns:
        extracted_watermark: Extracted binary sequence
        """
        # Read watermarked image
        watermarked_image = cv2.imread(watermarked_image_path)
        height, width = watermarked_image.shape[:2]

        # Extract from blue channel (least noticeable)
        extracted_watermark = []
        mid_band_coords = self._get_mid_band_coords()

        for i in range(0, height, self.block_size):
            for j in range(0, width, self.block_size):
                block = watermarked_image[i:i+self.block_size, j:j+self.block_size, 0].astype(float)
                dct_block = self._apply_dct_to_block(block)

                for coord in mid_band_coords:
                    extracted_bit = 1 if dct_block[coord] > 0 else 0
                    extracted_watermark.append(extracted_bit)

        # Reshape and save extracted watermark
        extracted_watermark = np.array(extracted_watermark)
        watermark_size = int(np.sqrt(len(extracted_watermark)))
        watermark_image = extracted_watermark[:watermark_size**2].reshape(watermark_size, watermark_size)

        # Scale up for better visibility
        watermark_image = cv2.resize(watermark_image.astype(np.uint8) * 255, (256, 256),
                                   interpolation=cv2.INTER_NEAREST)

        # Save extracted watermark
        cv2.imwrite(output_path, watermark_image)

        # Display extracted watermark
        plt.figure(figsize=(8, 8))
        plt.imshow(watermark_image, cmap='gray')
        plt.title('Extracted Watermark')
        plt.axis('off')
        plt.show()

        return extracted_watermark

def main():
    """Example usage with user interaction"""
    print("DCT Watermarking System")
    print("-" * 20)

    # Get input image
    input_image = "../base.jpeg"

    # Get watermark option
    watermark_path = "../watermark.jpeg"

    # Create watermarker instance
    watermarker = DCTWatermark(block_size=8, alpha=15)

    # Embed watermark
    print("\nEmbedding watermark...")
    watermarked_img = watermarker.embed(input_image, watermark_path)
    output_path = 'watermarked_' + input_image.split('/')[-1]
    cv2.imwrite(output_path, watermarked_img)
    print(f"Watermarked image saved as: {output_path}")

    # Extract watermark
    print("\nExtracting watermark...")
    extracted_watermark = watermarker.extract(output_path)
    print("Extracted watermark saved as: extracted_watermark.png")
    print("Extracted watermark has been displayed in a new window")

if __name__ == "__main__":
    main()
