# Generated by Django 4.1.3 on 2022-12-17 12:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('basicConcepts', '0004_post_resultpost'),
    ]

    operations = [
        migrations.AlterField(
            model_name='post',
            name='resultPost',
            field=models.CharField(blank=True, max_length=20, null=True),
        ),
    ]
