from django.db import models

# Create your models here.
class ImageDescription(models.Model):
    main_image_url = models.CharField(max_length=1000)
    

