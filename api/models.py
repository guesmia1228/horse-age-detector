import uuid
import random, string
from enum import Enum

from django.db import models
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
from django.contrib.auth.models import BaseUserManager
from django.utils.translation import ugettext_lazy as _
from django.core.mail import send_mail

import api.age_detector as agedetector


def gen_random_string(length):
    char_set = string.ascii_letters + string.digits
    if not hasattr(gen_random_string, "rng"):
        gen_random_string.rng = random.SystemRandom()  # Create a static variable
    return ''.join([gen_random_string.rng.choice(char_set) for _ in range(length)])


class PremiumType(Enum):
    Trial = 'trial'
    Monthly = 'monthly'
    Annually = 'annually'

    @classmethod
    def all(self):
        return [PremiumType.Trial, PremiumType.Monthly, PremiumType.Annually]


class UserManager(BaseUserManager):
    use_in_migrations = True

    def _create_user(self, email, password, **kwargs):
        email = self.normalize_email(email)
        is_staff = kwargs.pop('is_staff', False)
        is_superuser = kwargs.pop('is_superuser', False)
        user = self.model(
            email=email,
            is_active=True,
            is_staff=is_staff,
            is_superuser=is_superuser,
            **kwargs
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_user(self, email, password=None, **extra_fields):
        return self._create_user(email, password, **extra_fields)

    def create_superuser(self, email, password, **extra_fields):
        return self._create_user(email, password, is_staff=True, is_superuser=True, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin):
    date_joined = models.DateTimeField(auto_now_add=True)
    email = models.EmailField(max_length=255, unique=True)
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    is_upgraded = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    is_superuser = models.BooleanField(default=False)
    is_staff = models.BooleanField(default=False)
    is_social = models.BooleanField(default=False)
    is_premium = models.CharField(
        max_length=10,
        choices=[(tag.value, tag.name) for tag in PremiumType.all()],
        default='trial'
    )
    is_video = models.BooleanField(default=False)
    objects = UserManager()

    USERNAME_FIELD = 'email'

    class Meta:
        app_label = 'api'
        verbose_name = _('user')
        verbose_name_plural = _('users')

    def __str__(self):
        return self.email

    def get_short_name(self):
        """
        Returns the short name for the user.
        """
        return self.first_name

    def get_full_name(self):
        """
        Returns the first_name plus the last_name, with a space in between.
        """
        full_name = '%s %s' % (self.first_name, self.last_name)
        return full_name.strip()

    def email_user(self, subject, message, from_email=None, **kwargs):
        """
        Sends an email to this User.
        """
        send_mail(subject, message, from_email, [self.email], **kwargs)

    def save(self, password=None, *args, **kwargs):
        if not self.password:
            self.password = gen_random_string(10)

        is_social = kwargs.pop('is_social', False)
        super(User, self).save(*args, **kwargs)


class TeethDetectionModel(models.Model):
    user = models.ForeignKey('api.User', on_delete=models.CASCADE, related_name='owner')
    detect_file = models.CharField(null=True, blank=True, default="", max_length=1000)
    file = models.ImageField(upload_to='teeth_pics', null=True, blank=True)
    age = models.FloatField(null=True, blank=True, default=0.0)
    name = models.CharField(null=True, blank=True, default='Test Image.', max_length=255)
    image_type = models.CharField(max_length=10, blank=True, default='Test.jpeg')
    description = models.CharField(max_length=255, blank=True, default='It is detected age for uploaded image.')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'api'
        verbose_name = _('tooth')
        verbose_name_plural = _('teeth')
        ordering = ['-uploaded_at']

    def __str__(self):
        return self.user.email

    def agedetect(self):
        imgtype = self.image_type
        if imgtype == 'lower':
            imgpath = agedetector.detect_box_below(self.file.path)
            if imgpath is None:
                age = 0.0
            else:
                self.detect_file = '/' + imgpath
                age = agedetector.detect_age_below(imgpath)
        elif imgtype == 'upper':
            imgpath = agedetector.detect_box_upper(self.file.path)
            if imgpath is None:
                age = 0.0
            else:
                self.detect_file = '/' + imgpath
                age = agedetector.detect_age_upper(imgpath)
        else:
            imgpath = agedetector.detect_box_side(self.file.path)
            if imgpath is None:
                age = 0.0
            else:
                self.detect_file = '/' + imgpath
                age = agedetector.detect_age_side(imgpath)

        self.age = age
        self.image_type = imgtype

    def save(self, *args, **kwargs):
        super(TeethDetectionModel, self).save()
        self.agedetect()
        super(TeethDetectionModel, self).delete()
        super(TeethDetectionModel, self).save()


class Payment(models.Model):
    sender = models.ForeignKey('api.User', on_delete=models.CASCADE, related_name='sender')
    description = models.CharField(max_length=255, blank=True, default="It's test subscription.")
    created_at = models.DateTimeField(auto_now_add=True)
    amount = models.FloatField(null=True, blank=True, default=0.0)

    class Meta:
        app_label = 'api'

    def __str__(self):
        return self.sender.email


class Customer(models.Model):
    account = models.ForeignKey('api.User', on_delete=models.CASCADE, related_name='account')
    created_at = models.DateTimeField(auto_now_add=True)
    stripe_id = models.CharField(max_length=255, blank=True)
    sub_id = models.CharField(max_length=255, blank=True, default='')

    class Meta:
        app_label = 'api'

    def __str__(self):
        return self.account.email
