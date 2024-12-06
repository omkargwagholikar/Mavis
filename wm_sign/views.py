from django.shortcuts import render
from django.db import IntegrityError
import os
from django.conf import settings
from django.http import FileResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import UploadedImage
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from .signing_and_hashing import get_hash, encrypt_string, decrypt_string, public_key
from .watermarking import *

def watermark_image(file_path):
    pass

@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # image_file = request.FILES['image']
        image_file = request.FILES.get('image')

        file_path = os.path.join(settings.MEDIA_ROOT, 'uploads', image_file.name)

        with open(file_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)


        image_hash = get_hash(file_path=file_path)
        watermarked_image_path = embed_watermark(image_name=image_file.name, hash_value=image_hash)
        sign_hash = encrypt_string(image_hash)

        try:
            uploaded_image = UploadedImage(
                image='uploads/' + image_file.name,
                hash=image_hash
            )
            uploaded_image.save()

        except IntegrityError as e:
            return JsonResponse(data={'message':"The file seems to already exist in the database", "error":str(e)}, status=400)
        except Exception as e:
            return JsonResponse(data={
                    "message": str(e)
                },
                status=400
            )

        try:
            img_file = open(watermarked_image_path, 'rb')
            response = FileResponse(img_file, content_type='image/jpeg')
            response['Content-Disposition'] = f'attachment; filename="{image_file.name}"'
            response['signed_hash'] = sign_hash  # Additional header to include the signed hash
            return response

        except Exception as e:
            return JsonResponse({'message': str(e)}, status=500)

        # return JsonResponse({'message': 'Image uploaded successfully', 'signed_hash': sign_hash}, status=200)

    return JsonResponse({
        'error': 'Invalid request',
        'request_method': "Correct" if request.method == 'POST' else "incorrect",
        'file_exist_at_image': "Correct" if  request.FILES.get('image') else "incorrect",
    }, status=400)


def get_public_key(request):
    file_name = "public_key.pem"
    file_path = f"./media/keys/{file_name}"
    response = FileResponse(open(file_path, 'rb'), content_type='application/octet-stream')
    response['Content-Disposition'] = f'attachment; filename="{file_name}"'
    return response


def test(request):
    image_path = "temp2.png"
    original_hash = get_hash(f"./media/uploads/{image_path}")
    print(f"Original Image Hash: {original_hash}")

    watermarked_image_path = embed_watermark(image_name=image_path, hash_value=original_hash)

    extracted_hash = extract_watermark(watermarked_image_path)
    print(f"Extracted Watermark Hash: {extracted_hash}")

    if original_hash == extracted_hash:
        return JsonResponse({"message":"The watermark matches the original hash. Verification successful."})
    else:
        JsonResponse({"message":"The watermark does not match. Verification failed."})
