from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone


class CustomUser(AbstractUser):
    email = models.EmailField("email address", unique=True, blank=True)
    birthday = models.DateField("%m/%d/%y", default=timezone.now)
    is_active = models.BooleanField(default=False)
    activate_time = models.DateTimeField(default=None, null=True)
    activation_link_time = models.DateTimeField(default=None, null=True)
