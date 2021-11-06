from django.urls import path

from board import views

urlpatterns = [
    path("", views.MainView.as_view(), name="main"),
    path("<int:game_id>/detail/", views.DetailView.as_view(), name="detail"),
    path("detail/", views.FavouriteView.as_view(), name="edit"),
    path("history/", views.GetFavouriteView.as_view(), name="history"),
]
