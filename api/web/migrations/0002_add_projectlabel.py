# Generated by Django 3.1.3 on 2021-02-06 18:59

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('web', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ProjectLabel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('name', models.CharField(max_length=50)),
                ('description', models.TextField(default='New Label')),
                ('color', models.CharField(default='#000000', max_length=7)),
                ('project', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='web.project')),
            ],
            options={
                'abstract': False,
            },
        ),
    ]
