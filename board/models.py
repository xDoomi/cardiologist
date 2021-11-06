import uuid

from django.db import models

from users.models import CustomUser


class Favourite(models.Model):
    game_id = models.IntegerField()
    user = models.ForeignKey(
        CustomUser, on_delete=models.CASCADE, related_name="favourite_games"
    )


class Game(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=150, unique=True)
    slug = models.CharField(max_length=150, unique=True)
    full_description = models.TextField()
    release = models.DateTimeField(default=None, null=True)
    rating = models.FloatField(null=True)
    rating_count = models.IntegerField(null=True)
    aggregated_rating = models.FloatField(null=True)
    aggregated_rating_count = models.IntegerField(null=True)


class Image(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    url = models.CharField(max_length=150, unique=True)
    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name="images")
    is_cover = models.BooleanField(default=False)


class Genre(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=150, unique=True)
    game = models.ManyToManyField(Game, related_name="genres")


class Platform(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=150, unique=True)
    game = models.ManyToManyField(Game, related_name="platforms")
