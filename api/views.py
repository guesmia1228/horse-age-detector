import os
import smtplib
from email.mime.image import MIMEImage

import stripe
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.core.files.storage import FileSystemStorage
from django.shortcuts import get_object_or_404, render, redirect
from django.core.mail import EmailMessage, send_mail
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework import status, permissions
from rest_framework.response import Response
from rest_framework.authtoken.models import Token

from api.models import User, TeethDetectionModel, gen_random_string, Payment, Customer
from .serializers import UserSerializer, LoginUserSerializer, TeethDataSerializer, PasswordSerializer, PaymentSerializer

stripe.api_key = settings.STRIPE_LIVE_SECRET_KEY
# stripe.api_key = settings.STRIPE_TEST_SECRET_KEY


def parse_serializer_error(serializer):
    errors = serializer.errors
    error_keys = list(errors.keys())
    detail = ''
    for key in error_keys:
        for error in errors[key]:
            if len(detail) > 0:
                detail = detail + ' ' + str(error)
            else:
                detail = str(error)

    return detail


class SocialLoginView(APIView):
    authentication_classes = ()
    permission_classes = ()

    @staticmethod
    def post(request):
        is_social = bool(request.data.get('is_social'))
        email_address = request.data.get('email')
        serializer = UserSerializer(data=request.data, context={'request': request})
        if serializer.is_valid() and is_social:
            try:
                user = serializer.save()
                password = user.password
                # email = EmailMessage('Your account password', 'We generated password:' + str(password) + '.',
                #                      to=[email_address])
                # email.send()
                return Response(UserSerializer(user).data, status=status.HTTP_201_CREATED)
            except Exception as e:
                return Response(data={'message': str(e)},
                                status=status.HTTP_400_BAD_REQUEST)

        else:
            if is_social and User.objects.filter(email=email_address).exists():
                user = User.objects.get(email=email_address)
                return Response(UserSerializer(user).data, status=status.HTTP_200_OK)
            else:
                error_detail = parse_serializer_error(serializer)
                return Response(data={'message': error_detail}, status=status.HTTP_400_BAD_REQUEST)


class SignUpView(APIView):
    authentication_classes = ()
    permission_classes = ()

    @staticmethod
    def post(request):
        serializer = UserSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            try:
                password = request.data.get('password')
                user = serializer.save()
                user.password = password
                user.save()
                return Response(UserSerializer(user).data, status=status.HTTP_201_CREATED)
            except Exception as e:
                return Response(data={'message': 'There is unexpected error occurred.'},
                                status=status.HTTP_400_BAD_REQUEST)

        error_detail = parse_serializer_error(serializer)
        return Response(data={'message': error_detail}, status=status.HTTP_400_BAD_REQUEST)


class ForgotPasswordView(APIView):
    @staticmethod
    def post(request):
        email = request.data.get('email')
        try:
            user = User.objects.get(email=email)
            new_password = gen_random_string(10)
            user.password = new_password
            user.save()
            message = 'Hi ' + str(user.first_name) + ', your password was recently changed by the system. Your ' \
                                                     'changed password is ' + str(user.password) + '.'
            send_mail(
                'Your password was changed.',
                message,
                settings.EMAIL_HOST_USER,
                [email],
                fail_silently=False,
            )
            message = "We've changed your password successfully and sent new password to your email. Please check your email."
            return Response(data={'message': message}, status=status.HTTP_200_OK)
        except User.DoesNotExist:
            return Response(data={'message': 'There is no such user.'}, status=status.HTTP_400_BAD_REQUEST)
        except smtplib.SMTPException:
            message = 'There is the problem with mail server. Try again later.'
            return Response(data={'message': message}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response(data={'message': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class LoginView(APIView):
    authentication_classes = ()
    permission_classes = ()

    @staticmethod
    def post(request):
        email = request.data.get('email')
        password = request.data.get('password')
        try:
            user = User.objects.get(email=email)
            if user.password == password:
                serializer = LoginUserSerializer(user)
                data = {'id': user.id, 'email': user.email, 'first_name': user.first_name, 'last_name': user.last_name,
                        'is_social': user.is_social, 'is_premium': user.is_premium, 'is_video': user.is_video}
                if user.is_video:
                    payment = Payment.objects.filter(sender=user.id).filter(amount__gte=49.99)
                    if len(payment) > 0:
                        data['video_created_at'] = payment[0].created_at

                return Response(data=data, status=status.HTTP_200_OK)
            else:
                return Response(data={'message': 'Username or password is incorrect.'},
                                status=status.HTTP_400_BAD_REQUEST)
        except User.DoesNotExist:
            return Response(data={'message': 'There is no such user.'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response(data={'message': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class ProfileUpdateView(APIView):
    authentication_classes = ()
    permission_classes = ()

    @staticmethod
    def post(request):
        user_id = request.data.get('user_id')
        email = request.data.get('email')
        first_name = request.data.get('first_name')
        last_name = request.data.get('last_name')
        password = request.data.get('password')
        try:
            user1 = User.objects.get(id=user_id)
            if User.objects.filter(email=email).exists():
                user2 = User.objects.get(email=email)
                if user2 and user2.email != user1.email:
                    return Response(data={'message': 'There is already exist an user with requested email.'},
                                    status=status.HTTP_400_BAD_REQUEST)
                else:
                    user1.email = email
                    if first_name is not None and len(first_name) > 0:
                        user1.first_name = first_name
                    if last_name is not None and len(last_name) > 0:
                        user1.last_name = last_name
                    if password is not None and len(password) > 0:
                        user1.password = password
                    user1.save()
                    return Response(PasswordSerializer(user1).data, status=status.HTTP_200_OK)
            else:
                user1.email = email
                if first_name is not None and len(first_name) > 0:
                    user1.first_name = first_name
                if last_name is not None and len(last_name) > 0:
                    user1.last_name = last_name
                if password is not None and len(password) > 0:
                    user1.password = password
                user1.save()
                return Response(PasswordSerializer(user1).data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(data={'message': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class ChangePasswordView(APIView):
    authentication_classes = ()
    permission_classes = ()

    @staticmethod
    def post(request):
        email = request.data.get('email')
        old_password = request.data.get('old_password')
        new_password = request.data.get('new_password')
        try:
            user = User.objects.get(email=email)
            if user.password != old_password:
                return Response(data={'message': 'You entered wrong password for checking. Please try again.'},
                                status=status.HTTP_400_BAD_REQUEST)
            else:
                user.password = new_password
                user.save()
                return Response(PasswordSerializer(user).data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(data={'message': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class OneTimeChargeView(APIView):
    authentication_classes = ()
    permission_classes = ()

    @staticmethod
    def post(request):
        id = request.data.get('user')
        type = request.data.get('type')
        token = request.data.get('token')

        try:
            user = User.objects.get(id=id)
            if type == 'video':
                amount = 99.99
                description = "Payment for buying course videos."
            elif type == 'detection':
                amount = 7.99
                description = "Payment for buying one detection."
            else:
                return Response(data={'message': 'Requested wrong type.'}, status=status.HTTP_400_BAD_REQUEST)

            try:
                charge = stripe.Charge.create(
                    amount=int(amount * 100),
                    currency="usd", description=description, source=token)
            except stripe.error.CardError as e:
                return Response(data={'message': 'There is the problem with your card. Please check and try again.'},
                                status=status.HTTP_400_BAD_REQUEST)
            except stripe.error.RateLimitError as e:
                return Response(data={'message': 'There were too many requests hit the stripe API. Try again later.'},
                                status=status.HTTP_400_BAD_REQUEST)
            except stripe.error.InvalidRequestError as e:
                return Response(data={'message': 'Invalid request to sent to us. Please check your data again.'},
                                status=status.HTTP_400_BAD_REQUEST)
            except stripe.error.APIConnectionError as e:
                return Response(data={'message': "Failed to connect to Stripe. Try again later."},
                                status=status.HTTP_400_BAD_REQUEST)
            except stripe.error.SignatureVerificationError as e:
                return Response(data={'message': 'Failure to verify payload with signature attached to requests.'},
                                status=status.HTTP_400_BAD_REQUEST)
            except stripe.error.AuthenticationError as e:
                return Response(data={'message': 'Failed to properly authenticate yourself in the request.'},
                                status=status.HTTP_400_BAD_REQUEST)
            except stripe.error.APIError as e:
                return Response(data={'message': "API errors cover any other type of "
                                                 "problem (e.g., a temporary problem with Stripe’s servers) "
                                                 "and are extremely uncommon."},
                                status=status.HTTP_400_BAD_REQUEST)

            if type == 'video':
                user.is_video = True
                user.save()

            payment = Payment.objects.create(sender=user, description=description,
                                             amount=amount)
            data = {'id': user.id, 'email': user.email, 'first_name': user.first_name, 'last_name': user.last_name,
                    'is_social': user.is_social, 'is_premium': user.is_premium, 'is_video': user.is_video}
            if user.is_video:
                data['video_created_at'] = payment.created_at

                message = "Thank you for your interest in Aging Horses. We hope you enjoy this Age Old Skill as much as we do.\n" \
                          "If you have any questions, feel free to contact me at wayne@agemyhorse.com.\n" \
                          " <br>I’d be happy to help.</br>" \
                          "<br>Thanks again</br>" \
                          "<br>Wayne Needham.</br>"

                msg = EmailMessage('Congratulation Video Purchase', message, settings.EMAIL_HOST_USER, [user.email])
                msg.content_subtype = "html"
                msg.send()

                message = "Hi, Wayne. User " + user.get_full_name() + " just bought our video. His email is " + user.email + ". From CHAP"

                msg = EmailMessage('Alerts for Video Users', message, settings.EMAIL_HOST_USER, ['support@agemyhorse.com'])
                msg.content_subtype = "html"
                msg.send()
            return Response(data, status=status.HTTP_200_OK)
        except User.DoesNotExist:
            return Response(data={'message': 'There is no such user.'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response(data={'message': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class UpgradeView(APIView):
    authentication_classes = ()
    permission_classes = ()

    @staticmethod
    def post(request):
        email = request.data.get('email')
        stripe_token = request.data.get('token')
        premium = request.data.get('subscription')

        try:
            user = User.objects.get(email=email)
            stripe_customer = stripe.Customer.create(source=stripe_token, email=email,
                                                     description='Customer for monthly subscription.')
            customer_id = stripe_customer['id']
            if premium == 'monthly':
                subscription = stripe.Subscription.create(customer=customer_id,
                                                      items=[{'plan': settings.STRIPE_LIVE_MONTHLY_PLAN_ID, 'quantity': 1, }, ])
            elif premium == 'annually':
                subscription = stripe.Subscription.create(customer=customer_id,
                                                      items=[{'plan': settings.STRIPE_LIVE_ANNUALLY_PLAN_ID, 'quantity': 1, }, ])
            sub_id = subscription['id']
            customer = Customer.objects.create(account=user, stripe_id=customer_id, sub_id=sub_id)
            user.is_premium = premium
            user.save()

            message = "Hi, Wayne. User " + user.get_full_name() + " just upgraded as premium user. His email is " + email + ". From CHAP"

            msg = EmailMessage('Alerts for Premium Users', message, settings.EMAIL_HOST_USER, ['support@agemyhorse.com'])
            msg.content_subtype = "html"
            msg.send()

            message = "Thank you for using CHAP as your Horse Aging Service. We hope you will find our App easy to use and our service even better.\n" \
                      "Feel free to contact me at wayne@agemyhorse.com if you have any questions or concerns.\n" \
                      "<br>Thanks again</br>" \
                      "<br>Wayne Needham.</br>"

            msg = EmailMessage('Congratulate Premium User', message, settings.EMAIL_HOST_USER, [email])
            msg.content_subtype = "html"
            msg.send()

            return Response(UserSerializer(user).data, status.HTTP_200_OK)
        except User.DoesNotExist:
            return Response(data={'message': 'There is no such user.'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response(data={'message': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class DowngradeView(APIView):
    authentication_classes = ()
    permission_classes = ()

    @staticmethod
    def post(request):
        email = request.data.get('email')
        try:
            user = User.objects.get(email=email)
            customer = Customer.objects.get(account=user)
            customer_id = customer.stripe_id
            sub_id = customer.sub_id
            customer_delete = stripe.Customer.delete(customer_id)
            # sub_delete = stripe.Subscription.delete(sub_id)
            customer.delete()
            user.is_premium = 'trial'
            user.save()

            message = "Hi, Wayne. {} ({}) just unsubscribed.".format(user.get_full_name(), user.email)
            msg = EmailMessage('CHAP User Unsubscribed', message, settings.EMAIL_HOST_USER, ['support@agemyhorse.com'])
            msg.content_subtype = "html"
            msg.send()

            return Response(UserSerializer(user).data, status.HTTP_200_OK)
        except User.DoesNotExist:
            return Response(data={'message': 'There is no such user.'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response(data={'message': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class SearchByNameView(APIView):
    @staticmethod
    def post(request):
        email = request.data.get('email')
        search_name = request.data.get('search_name')


class LogoutView(APIView):
    @staticmethod
    def get(request):
        token = get_object_or_404(Token, key=request.auth)
        token.delete()
        return Response(data={'message': 'Successfully logout.'}, status=status.HTTP_200_OK)


class GetAllUserView(APIView):

    @staticmethod
    def get(request):
        users = User.objects.all()
        return Response(UserSerializer(users, many=True).data, status=status.HTTP_200_OK)


def home(request):
    return render(request, 'home.html')


def login(request):
    return render(request, 'home.html')


def register(request):
    return render(request, 'home.html')


class DetectView(APIView):
    @staticmethod
    def post(request):
        serializer = TeethDataSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            try:
                file = request.data.get('file')
                if file is not None:
                    user = request.data.get('user')
                    teeth_data = serializer.save()
                    all_teeths = TeethDetectionModel.objects.filter(user=user)
                    data = {'recent': TeethDataSerializer(teeth_data).data,
                            'all': TeethDataSerializer(all_teeths, many=True).data}
                    return Response(data=data, status=status.HTTP_200_OK)
                else:
                    return Response(data={'message': 'There is no any image file.'}, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                return Response(data={'message': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        else:
            error_detail = parse_serializer_error(serializer)
            return Response(data={'message': error_detail}, status=status.HTTP_400_BAD_REQUEST)

    @staticmethod
    def get(request, *args, **kwargs):
        try:
            userId = kwargs['pk']
            all_teeths = TeethDetectionModel.objects.filter(user=userId)
            data = TeethDataSerializer(all_teeths, many=True).data
            return Response(data=data, status=status.HTTP_200_OK)
        except:
            return Response(data={'message': 'Something went wrong.'}, status=status.HTTP_400_BAD_REQUEST)


class AnswerView(APIView):
    # permission_classes = (permissions.IsAuthenticated,)

    @staticmethod
    def post(request):
        userId = request.data.get('user')
        detectId = request.data.get('detection')
        user_age = request.data.get('age')
        try:
            user = User.objects.get(id=userId)
            detect = TeethDetectionModel.objects.get(id=detectId)

            detect_path = detect.detect_file[1:]
            folder_name = os.path.basename(os.path.dirname(detect_path))
            basename = os.path.basename(detect_path)
            fs = FileSystemStorage()
            filePath = fs.path(folder_name + '/' + basename)

            message = "Thank you for using CHAP Horse Aging Service. Our Expert staff should be in contact in the next 24 hours to further assist you.\n" \
                      "We hope you had a great experience. If you get the desire to learn how to Age Horses, we offer a comprehensive Aging Course guaranteed to get you Aging your own horses fast and easy.\n" \
                      "We also offer a monthly Subscription to allow you to use our service as often as you like.\n" \
                      "Below is a copy of your image and our analysis. If you have any questions, feel free to contact me at wayne@agemyhorse.com.\n" \
                      "<br>Thanks again for your patronage</br>" \
                      "<br>Wayne Needham.</br>"

            msg = EmailMessage('Horse Age Detection', message, settings.EMAIL_HOST_USER, [user.email])
            msg.content_subtype = "html"
            msg.attach_file(detect.file.path)
            msg.attach_file(filePath)
            msg.send()

            if user.is_premium == 'monthly':
                message = "Hi, we've detected a horse age from " + str(
                    user.first_name) + ". \nUser suggestion is " + str(user_age) + ". \nDetected age is " + str(
                    detect.age) + ". \nAlso there are two images: First one is from the user and second one is result" \
                                ". \nUser's email is " + user.email + ". This is Premium User on $9.99 Subscription."
            elif user.is_premium == 'annually':
                message = "Hi, we've detected a horse age from " + str(
                    user.first_name) + ". \nUser suggestion is " + str(user_age) + ". \nDetected age is " + str(
                    detect.age) + ". \nAlso there are two images: First one is from the user and second one is result" \
                                ". \nUser's email is " + user.email + ". This is Premium User on $99.99 Subscription."
            else:
                message = "Hi, we've detected a horse age from " + str(
                    user.first_name) + ". \nUser suggestion is " + str(user_age) + ". \nDetected age is " + str(
                    detect.age) + ". \nAlso there are two images: First one is from the user and second one is result" \
                                ". \nUser's email is " + user.email + ". This is General User on $7.99 Detection"

            msg = EmailMessage('Horse Age Detection', message, settings.EMAIL_HOST_USER, ['support@agemyhorse.com'])
            msg.content_subtype = "html"
            msg.attach_file(detect.file.path)
            msg.attach_file(filePath)
            msg.send()
            return Response(data=TeethDataSerializer(detect).data, status=status.HTTP_200_OK)
        except Exception as e:
            print(str(e))
            return Response(data={'message': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class DetectionList(APIView):
    @staticmethod
    def get(request, user_id):
        try:
            all_teeths = TeethDetectionModel.objects.filter(user=user_id)
            data = TeethDataSerializer(all_teeths, many=True).data
            return Response(data=data, status=status.HTTP_200_OK)
        except:
            return Response(data={'message': 'Something went wrong.'}, status=status.HTTP_400_BAD_REQUEST)
