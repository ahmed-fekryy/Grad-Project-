# Generated by Django 4.1.3 on 2022-12-17 12:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('basicConcepts', '0003_remove_post_resultchest_remove_post_user'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='resultPost',
            field=models.CharField(default=1, max_length=20),
            preserve_default=False,
        ),
    ]
