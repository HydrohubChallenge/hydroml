from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.db import models
from colorful.fields import RGBColorField

import pandas as pd
import os


def user_directory_path(instance, filename):
    # File will be uploaded to MEDIA_ROOT/datasets/user_<id>/filename
    return 'datasets/user_{0}/{1}'.format(instance.owner_id, filename)


class BaseModel(models.Model):
    created_at = models.DateTimeField(
        auto_now_add=True,
    )

    updated_at = models.DateTimeField(
        auto_now=True,
    )

    class Meta:
        abstract = True


class Project(BaseModel):
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name='projects',
        on_delete=models.DO_NOTHING,
        null=True,
        blank=True,
    )

    name = models.CharField(
        max_length=50,
    )

    description = models.TextField(
        default='New Project',
    )

    # Dataset as blob. Future implementation, if needed.
    # dataset = models.BinaryField(
    #     blank=True,
    #     null=True,
    #     editable=True,
    # )

    # Dataset as File.
    dataset = models.FileField(
        null=True,
        upload_to=user_directory_path,
    )

    delimiter = models.CharField(
        max_length=3,
        default=',',
    )


    def __str__(self):
        return self.name


class Label(BaseModel):
    project = models.ForeignKey(
        Project,
        on_delete=models.DO_NOTHING,
    )

    name = models.CharField(
        max_length=50,
    )

    description = models.TextField(
        default='New Label',
    )

    color = models.CharField(
        max_length=7,
        default='#000000',
    )


    def __str__(self):
        return self.name