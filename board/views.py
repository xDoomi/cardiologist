import json

from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.http import HttpResponse
from django.shortcuts import render
from django.views import View


class Filter:
    def __init__(self):
        params = {"fields": "name"}
        res_genres = None   # TODO
        res_platforms = None   # TODO

        self.genres = res_genres
        if res_genres:
            self.genres.insert(0, {"id": 0, "name": "Any"})
        self.platforms = res_platforms


class DetailView(View):
    def get(self, request, game_id):
        game = self._get_game(game_id)
        tweets = self._get_tweets(game.slug)
        user = request.user

        if user.is_authenticated:
            is_favourite = user.favourite_games.filter(
                game_id=self.kwargs["game_id"]
            ).exists()
        else:
            is_favourite = None
        context = {
            "game": game,
            "tweets": tweets,
            "is_favourite": is_favourite,
        }
        return render(request, "board/detail.html", context=context)

    @staticmethod
    def _get_game(game_id):
        return None   # TODO

    @staticmethod
    def _get_tweets(game_slug):
        return None   # TODO


class FavouriteView(View):
    def _data_init(self):
        data = json.loads(self.request.body.decode())
        self.game_id = data["game_id"]
        self.favourite_games = self.request.user.favourite_games

    def post(self, request):
        self._data_init()
        request.user.favourite_games.update_or_create(
            game_id=self.game_id, user=request.user
        )
        return HttpResponse(status=200)

    def delete(self, request):
        self._data_init()
        favourite_game = request.user.favourite_games.filter(
            game_id=self.game_id
        ).first()
        if favourite_game:
            favourite_game.delete()
        return HttpResponse(status=200)


class GetFavouriteView(View):
    @staticmethod
    def get(request):
        game_list = None   # TODO
        context = {
            "games": game_list,
        }
        return render(request, "board/favourite.html", context)


class MainView(View):
    def _data_init(self):
        data = self.request.GET
        self.platforms = data.getlist("platforms")
        self.genres = data.getlist("genres")
        self.rating = data.get("rating", default=50)

    def get(self, request):
        self._data_init()
        games = []   # TODO

        paginator = Paginator(games, 8)  # object_list
        page_number = request.GET.get("page")
        try:
            page_obj = paginator.get_page(page_number)
        except PageNotAnInteger:
            # Set first page
            page_obj = paginator.page(1)
        except EmptyPage:
            # Set last page, if the counter is bigger then max_page
            page_obj = paginator.page(paginator.num_pages)

        filter_panel = Filter()
        filter_initials = {
            "platforms": self.platforms,
            "genres": self.genres,
            "rating": int(self.rating),
        }
        context = {
            "games": games,
            "filter_panel": filter_panel,
            "filter_initials": filter_initials,
            "page_obj": page_obj,
            "page_numbers": paginator.page_range,
            "user": request.user,
        }
        return render(request, "board/main.html", context=context)
