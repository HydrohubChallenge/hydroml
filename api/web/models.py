from django.conf import settings
from django.db import models


class BaseModel(models.Model):
    created_at = models.DateTimeField(
        auto_now_add=True,
    )

    updated_at = models.DateTimeField(
        auto_now_add=True,
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

    describe = models.TextField(
        default='New Project',
    )

    def __str__(self):
        return self.name
