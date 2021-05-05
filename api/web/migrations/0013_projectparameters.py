# Generated by Django 3.1.3 on 2021-05-03 17:44

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('web', '0012_remove_projectprediction_confusion_matrix'),
    ]

    operations = [
        migrations.CreateModel(
            name='ProjectParameters',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('name', models.CharField(max_length=50)),
                ('value', models.IntegerField(null=True)),
                ('project', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='web.project')),
            ],
            options={
                'abstract': False,
            },
        ),
    ]