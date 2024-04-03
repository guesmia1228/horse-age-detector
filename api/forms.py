from django import forms


class UserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput())
    confirm_password = forms.CharField(widget=forms.PasswordInput())

    class Meta:
        fields = ('first_name', 'last_name', 'email', 'avatar', 'password')

    def password_valid(self):
        cleaned_data = super(UserForm, self).clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')

        if password != confirm_password:
            return False
        else:
            return True


class LoginForm(forms.ModelForm):
    class Meta:
        fields = ('email', 'password')
