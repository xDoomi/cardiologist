from django.urls import path

from users import views as users_views

urlpatterns = [
    path("signup/", users_views.SignUpView.as_view(), name="signup"),
    path(
        "reactivate/<slug:user_id>/", users_views.resend_auth_mail, name="reactivation"
    ),
    path("activate/<slug:user_id>/", users_views.account_activation, name="activation"),
    path("profile/", users_views.get_user_profile, name="profile"),
]
