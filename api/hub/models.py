from django.db import models

class Project(models.Model):
    name = models.CharField(max_length=50)
    describe = models.TextField(default='New Project')
    def __str__(self):
        return self.name