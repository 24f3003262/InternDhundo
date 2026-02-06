from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.contrib import messages
from .models import Application
import pandas as pd  # Import pandas

# Import your AppConfig to access the globally loaded AI model
from .apps import InterndhundoConfig  # ❗️IMPORTANT: Replace with your actual AppConfig name

def index(request):
    return render(request, "index.html")

def login_page(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}!")
            return redirect("dashboard")
        else:
            messages.error(request, "Invalid username or password")
            # The original code had a path here that might be incorrect.
            # Pointing to "login.html" is safer.
            return render(request, "login.html")
    return render(request, "login.html")

@login_required(login_url="login")
def dashboard(request):
    # --- MODIFICATION START ---
    # Retrieve recommendations from the session after a redirect
    recommendations_data = request.session.pop('recommendations', None) # .pop() removes it after retrieval
    latest_application = Application.objects.filter(user=request.user).order_by('-id').first()

    context = {
        'recommendations': recommendations_data,
        'application': latest_application,
    }
    return render(request, "dashboard.html", context)
    # --- MODIFICATION END ---

@login_required(login_url="login")
def application_page(request):
    return render(request, "application.html")

@login_required(login_url="login")
def submit_application(request):
    if request.method == "POST":
        # Create and save the application to the database
        application = Application.objects.create(
            user=request.user,  # Link to the logged-in user
            full_name=request.POST.get("fullName"),
            email=request.POST.get("email"),
            phone=request.POST.get("phone"),
            dob=request.POST.get("dob"),
            address=request.POST.get("address"),
            gender=request.POST.get("gender"),
            marital_status=request.POST.get("maritalStatus"),
            nationality=request.POST.get("nationality"),
            linkedin=request.POST.get("linkedin"),
            father_name=request.POST.get("fatherName"),
            mother_name=request.POST.get("motherName"),
            religion=request.POST.get("religion"),
            languages=request.POST.get("languages"),
            degree=request.POST.get("degree"),
            interested_roles=request.POST.get("interestedRoles"),
            achievements=request.POST.get("achievements"),
            preference=request.POST.get("preference"),
            city=request.POST.get("city"),
            experience=request.POST.get("experience"),
            skills=request.POST.get("skills"),
        )
        messages.success(request, "Application submitted successfully! See your AI recommendations below.")
        
        # --- MODIFICATION START: Run the AI Model ---
        recommendations = []
        if InterndhundoConfig.matcher is not None:
            matcher = InterndhundoConfig.matcher
            weights = InterndhundoConfig.optimal_weights
            nlp = InterndhundoConfig.nlp_model
            
            # Prepare the user profile from the submitted application
            user_profile = pd.Series({
                'interested_roles': application.interested_roles,
                'skillsets': application.skills, # Ensure 'skills' from your form matches this
                'experience': application.experience,
                'achievements': application.achievements
            })
            
            # Get recommendations
            recommendations_df = matcher.match(
                user_profile,
                nlp=nlp,
                top_n=5,
                tfidf_weight=weights[0],
                semantic_weight=weights[1]
            )
            filtered_df = recommendations_df[recommendations_df['hybrid_score'] >= 0.5]
            recommendations = filtered_df.to_dict('records')
        else:
            messages.warning(request, "AI model not loaded. Cannot provide recommendations.")

        # Store recommendations in the session to pass them to the dashboard
        request.session['recommendations'] = recommendations
        
        # --- MODIFICATION END ---
        
        return redirect("dashboard")

    return HttpResponse("Invalid request method", status=405)

def register(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirmPassword")

        if password != confirm_password:
            messages.error(request, "Passwords do not match")
            return redirect("register")

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
            return redirect("register")

        user = User.objects.create_user(username=username, email=email, password=password)
        messages.success(request, "Account created successfully! Please log in.")
        return redirect("login")

    return render(request, "register.html")