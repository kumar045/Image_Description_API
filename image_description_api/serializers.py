from rest_framework import serializers


class ImageDescriptionSerializer(serializers.Serializer):
    main_image_url = serializers.CharField()
    

