import csv

from django.contrib import admin
from django.utils.translation import ugettext_lazy as _
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.http import HttpResponse

from api.models import User, TeethDetectionModel, Payment, Customer


class CustomUserAdmin(BaseUserAdmin):
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        (_('Personal info'), {'fields': ('first_name', 'last_name')}),
        (_('Permissions'), {'fields': ('is_active', 'is_staff', 'is_superuser')}),
        (_('Video Purchased'), {'fields': ('is_video',)}),
        (_('Membership Upgraded'), {'fields': ('is_premium',)}),
        # (_('Important dates'), {'fields': ('date_joined',)}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password1', 'password2'),
        }),
    )
    readonly_fields = ("date_joined",)
    list_display = ('email', 'first_name', 'last_name', 'is_superuser', 'is_staff')
    search_fields = ('email', 'first_name', 'last_name')
    ordering = ('email',)
    filter_horizontal = ()

    def export_users(self, request, queryset):
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="users.csv"'

        writer = csv.writer(response)
        writer.writerow(['Email', 'First Name', 'Last Name', 'Is Video', 'Membership'])

        for user in queryset:
            row = [user.email, user.first_name, user.last_name, user.is_video, user.is_premium]
            writer.writerow(row)

        return response

    actions = [
        export_users
    ]


class TeethDetectionAdmin(admin.ModelAdmin):
    list_display = ('user', 'name', 'uploaded_at')


admin.site.register(User, CustomUserAdmin)
admin.site.register(TeethDetectionModel, TeethDetectionAdmin)
admin.site.register(Payment)
admin.site.register(Customer)
