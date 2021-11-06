from datetime import timedelta

from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render
from django.template.loader import render_to_string
from django.urls import reverse_lazy
from django.utils import timezone
from django.views import generic

from cardiologist.settings import ACCOUNT_ACTIVATION_URL

from .forms import CustomUserCreationForm
from .models import CustomUser


class SignUpView(generic.CreateView):
    form_class = CustomUserCreationForm
    success_url = reverse_lazy("login")
    template_name = "users/signup.html"

    def form_valid(self, form):
        super().form_valid(form)
        form = AuthenticationForm()
        return render(
            self.request,
            "registration/login.html",
            context={"nextstep": True, "form": form},
        )


def account_activation(request, user_id):
    user = CustomUser.objects.get(id=user_id)
    delta = timedelta(hours=3)
    difference = timezone.now() - user.activation_link_time
    context = {
        "title": "Something goes wrong...",
        "text": "Your activation link has expired. "
        "Click on the button to create a new activation email.",
        "btn_name": "create email",
        "btn_url_name": "reactivation",
        "user": user,
    }
    if difference < delta:
        if user.is_active:
            context = {
                "title": "DONE!",
                "text": "Your account has already activated.",
                "btn_name": "to site",
                "btn_url_name": "main",
            }
        else:
            user.is_active = True
            user.activate_time = timezone.now()
            user.save()
            context = {
                "title": "Congratulations!",
                "text": "You have activated you account.",
                "btn_name": "to site",
                "btn_url_name": "main",
            }
    return render(request, "users/activation.html", context=context)


def resend_auth_mail(request, user_id):
    user = CustomUser.objects.get(id=user_id)
    now = timezone.now()
    user.activation_link_time = now
    user.save()
    url = f"{ACCOUNT_ACTIVATION_URL}{user.id}/"
    context = {
        "username": user.username,
        "activation_link": url,
    }
    html_body = render_to_string("users/email.html", context)
    user.email_user("Account activation", "Confirmation", html_message=html_body)

    return render(request, "registration/login.html", context={"nextstep": True})


def get_user_profile(request):
    user = request.user
    user_data = {
        "Username": user.username,
        "Email": user.email,
        "Birthday": user.birthday,
        "Password": None,
    }
    context = {
        "user": user,
        "user_data": user_data,
        "next": "password_change_done",
    }
    return render(request, "users/profile.html", context)
