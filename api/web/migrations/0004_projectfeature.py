# Generated by Django 3.1.3 on 2021-02-06 19:19

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('web', '0003_add_projectprediction'),
    ]

    operations = [
        migrations.CreateModel(
            name='ProjectFeature',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('column', models.CharField(max_length=50)),
                ('type', models.IntegerField(choices=[(1, 'Target'), (2, 'Skip'), (3, 'Input'), (4, 'Timestamp')], default=2)),
                ('project', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='web.project')),
            ],
            options={
                'abstract': False,
            },
        ),
    ]
