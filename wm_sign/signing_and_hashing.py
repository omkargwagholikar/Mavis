import hashlib
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
import base64
import os

from cryptography.hazmat.primitives import serialization

# Load the private key from the file
with open("./media/keys/private_key.pem", "rb") as private_file:
    private_key = serialization.load_pem_private_key(
        private_file.read(), password=None, backend=default_backend()
    )

# Load the public key from the file
with open("./media/keys/public_key.pem", "rb") as public_file:
    public_key = serialization.load_pem_public_key(
        public_file.read(), backend=default_backend()
    )


def encrypt_string(plain_text: str) -> str:
    """Encrypts a given string using the RSA public key."""
    plain_text_bytes = plain_text.encode("utf-8")

    # Encrypt the bytes
    encrypted_bytes = public_key.encrypt(
        plain_text_bytes,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    # Encode the encrypted bytes in base64 to make it string-friendly
    encrypted_base64 = base64.b64encode(encrypted_bytes)
    return encrypted_base64.decode("utf-8")


def decrypt_string(encrypted_text: str) -> str:
    """Decrypts a given encrypted string using the RSA private key."""
    encrypted_bytes = base64.b64decode(encrypted_text)
    decrypted_bytes = private_key.decrypt(
        encrypted_bytes,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return decrypted_bytes.decode("utf-8")


# Function to get the SHA-256 hash of an image file
def get_hash(file_path):
    # Initialize SHA-256 hash object
    digest = hashlib.sha256()

    # Read the image file in binary mode and update the hash
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            digest.update(byte_block)

    # Return the hash as a hexadecimal string
    return digest.hexdigest()
