from rest_framework import serializers
from rest_framework.authtoken.models import Token
from django.contrib.auth.password_validation import validate_password

from .models import User, TeethDetectionModel, Customer, Payment


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'email', 'first_name', 'last_name', 'is_social', 'is_premium', 'is_video')


class PasswordSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'email', 'first_name', 'last_name', 'is_social', 'is_premium', 'is_video', 'password')


class LoginUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'email', 'first_name', 'last_name', 'is_social', 'is_premium', 'is_video')


class TeethDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = TeethDetectionModel
        fields = ('id', 'user', 'detect_file', 'file', 'age', 'image_type', 'description', 'uploaded_at', 'name')


class CustomerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Customer
        fields = ('account', 'stripe_id', 'created_at')


class PaymentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Payment
        fields = ('sender', 'created_at', 'description', 'amount')
