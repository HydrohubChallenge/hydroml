# Generated by Django 3.1.3 on 2021-04-03 18:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('web', '0008_apidata'),
    ]

    operations = [
        migrations.AlterField(
            model_name='apidata',
            name='result',
            field=models.IntegerField(null=True),
        ),
    ]
