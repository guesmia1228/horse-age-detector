from django.contrib import admin
from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static

from api.views import *

admin.site.site_header = "Age Horse Service Admin"

urlpatterns = [
    url(r'^api/register$', SignUpView.as_view()),
    url(r'^api/login$', LoginView.as_view()),
    url(r'^api/sociallogin$', SocialLoginView.as_view()),
    url(r'^api/forgotpassword$', ForgotPasswordView.as_view()),
    url(r'^api/changeprofile$', ProfileUpdateView.as_view()),
    url(r'^api/changepassword$', ChangePasswordView.as_view()),
    url(r'^api/upgrade$', UpgradeView.as_view()),
    url(r'^api/downgrade$', DowngradeView.as_view()),
    url(r'^api/users$', GetAllUserView.as_view()),
    url(r'^api/logout$', LogoutView.as_view()),
    url(r'^api/detection$', DetectView.as_view()),
    url(r'^api/charge$', OneTimeChargeView.as_view()),
    url(r'^api/answer$', AnswerView.as_view()),
    url(r'^api/detections/(?P<user_id>[\d]+)$', DetectionList.as_view()),

    url(r'^$', home, name='home'),
    url(r'^login$', login, name='login'),
    url(r'^register$', register, name='register'),

    url(r'^admin/', admin.site.urls)
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
