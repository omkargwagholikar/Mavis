# image_upload/urls.py
from django.urls import path
from .views import upload_image, get_public_key, test

urlpatterns = [
    path("upload/", upload_image, name="upload_image"),
    path("get_public_key/", get_public_key, name="get_public_key"),
    path("test", test, name="test"),
]
