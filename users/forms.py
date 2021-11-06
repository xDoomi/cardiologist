from django.contrib.admin import widgets
from django.contrib.auth.forms import UserChangeForm, UserCreationForm
from django.template.loader import render_to_string
from django.utils import timezone

from cardiologist.settings import ACCOUNT_ACTIVATION_URL

from .models import CustomUser


class CustomUserCreationForm(UserCreationForm):
    class Meta(UserCreationForm):
        model = CustomUser
        fields = ("username", "email", "birthday")

    def __init__(self, *args, **kwargs):
        super(CustomUserCreationForm, self).__init__(*args, **kwargs)
        self.fields["birthday"].widget = widgets.AdminDateWidget()

    def save(self):
        user = super().save()
        self.send_auth_mail(user)
        return user

    def send_auth_mail(self, user):
        context = {
            "username": user.username,
            "activation_link": self._build_activation_link(user),
        }
        html_body = render_to_string("users/email.html", context)
        user.email_user("Account activation", "Confirmation", html_message=html_body)

    @staticmethod
    def _build_activation_link(user):
        now = timezone.now()
        user.activation_link_time = now
        user.save()
        url = f"{ACCOUNT_ACTIVATION_URL}{user.id}/"
        return url


class CustomUserChangeForm(UserChangeForm):
    class Meta:
        model = CustomUser
        fields = ("username", "email")
