from django.conf import settings
from django.db import models
from django.utils.translation import gettext_lazy as _


def user_directory_path(instance, filename):
    # File will be uploaded to MEDIA_ROOT/datasets/user_<id>/filename
    return 'datasets/user_{0}/{1}'.format(instance.owner_id, filename)


def serialized_prediction_directory_path(instance, filename):
    # File will be uploaded to MEDIA_ROOT/datasets/user_<id>/filename
    return 'models/project_{0}/{1}'.format(instance.project_id, filename)


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
    class Type(models.IntegerChoices):
        RAINFALL = 1, _('Rainfall')
        WATER_LEVEL = 2, _('Water Level')

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

    type = models.IntegerField(
        choices=Type.choices,
        null=True,
        blank=False,
    )

    dataset_file = models.FileField(
        null=True,
        upload_to=user_directory_path,
    )

    delimiter = models.CharField(
        max_length=3,
        default=',',
    )

    def __str__(self):
        return self.name


class ProjectLabel(BaseModel):
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


class ProjectPrediction(BaseModel):
    class StatusType(models.IntegerChoices):
        SUCCESS = 1, _('Success')
        TRAINING = 2, _('Training')
        ERROR = 3, _('Error')

    project = models.ForeignKey(
        Project,
        on_delete=models.DO_NOTHING,
    )

    status = models.IntegerField(
        choices=StatusType.choices,
        null=True,
        blank=True,
    )

    accuracy = models.DecimalField(
        blank=True,
        max_digits=18,
        decimal_places=17,
    )

    precision = models.DecimalField(
        blank=True,
        max_digits=18,
        decimal_places=17,
    )

    recall = models.DecimalField(
        blank=True,
        max_digits=18,
        decimal_places=17,
    )

    f1_score = models.DecimalField(
        blank=True,
        max_digits=18,
        decimal_places=17,
    )

    confusion_matrix = models.TextField(
        blank=True,
    )

    serialized_prediction_file = models.FileField(
        null=True,
        upload_to=serialized_prediction_directory_path,
    )

    def __str__(self):
        return f'{self.project} - {self.id}'


class ProjectFeature(BaseModel):
    class Type(models.IntegerChoices):
        TARGET = 1, _('Target')
        SKIP = 2, _('Skip')
        INPUT = 3, _('Input')
        TIMESTAMP = 4, _('Timestamp')

    project = models.ForeignKey(
        Project,
        on_delete=models.DO_NOTHING,
    )

    column = models.CharField(
        max_length=50,
    )

    type = models.IntegerField(
        choices=Type.choices,
        default=Type.SKIP
    )

    def __str__(self):
        return f'{self.project} - {self.column}'
