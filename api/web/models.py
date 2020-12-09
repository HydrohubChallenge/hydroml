from django.conf import settings
from django.db import models


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

    dataset = models.BinaryField(
        blank=True,
        null=True,
        editable=True,
    )

    def __str__(self):
        return self.name


class Label(BaseModel):
    project_id = models.ForeignKey(
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