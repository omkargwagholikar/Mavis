from django.db import models


# Create your models here.
class UploadedImage(models.Model):
    image = models.ImageField(upload_to="uploads/")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    hash = models.CharField(primary_key=True, max_length=1000)

    def __str__(self):
        return f"{self.image.name} - {self.hash}"
