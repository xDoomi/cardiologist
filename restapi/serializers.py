from rest_framework import serializers

from board.models import Game, Genre, Platform
from users.models import CustomUser as User


class GameSerializer(serializers.HyperlinkedModelSerializer):
    genres = serializers.PrimaryKeyRelatedField(many=True, queryset=Genre.objects.all())
    platforms = serializers.PrimaryKeyRelatedField(
        many=True, queryset=Platform.objects.all()
    )
    images = serializers.SlugRelatedField(many=True, read_only=True, slug_field="url")

    class Meta:
        model = Game
        fields = [
            "id",
            "name",
            "slug",
            "genres",
            "platforms",
            "full_description",
            "release",
            "rating",
            "rating_count",
            "aggregated_rating",
            "aggregated_rating_count",
            "images",
        ]


class GenreSerializer(serializers.HyperlinkedModelSerializer):
    game = serializers.PrimaryKeyRelatedField(many=True, read_only=True)

    class Meta:
        model = Genre
        fields = ["id", "name", "game"]


class PlatformSerializer(serializers.HyperlinkedModelSerializer):
    game = serializers.PrimaryKeyRelatedField(many=True, read_only=True)

    class Meta:
        model = Platform
        fields = ["id", "name", "game"]


class UserSerializer(serializers.ModelSerializer):
    is_staff = serializers.ReadOnlyField()
    last_login = serializers.ReadOnlyField()

    class Meta:
        model = User
        fields = ["id", "username", "email", "birthday", "is_staff", "last_login"]
