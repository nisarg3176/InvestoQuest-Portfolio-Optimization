# optimizer/urls.py

from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    # Authentication URLs
    path('login/', auth_views.LoginView.as_view(template_name='optimizer/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),

    # Main application URLs
    path('', views.welcome_view, name='welcome'), # Default page
    path('welcome/', views.welcome_view, name='welcome'),
    path('know-the-models/', views.know_the_models_view, name='know_the_models'),
    path('portfolio-optimizer/', views.portfolio_optimizer_view, name='portfolio_optimizer'),
    path('about-us/', views.about_us_view, name='about_us'),
    path('contact-us/', views.contact_us_view, name='contact_us'),
]