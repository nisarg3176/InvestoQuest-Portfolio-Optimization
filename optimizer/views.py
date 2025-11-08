# optimizer/views.py

from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import pandas as pd
from .forms import PortfolioOptimizerForm # <-- UPDATED
from optimizer.models.mean_variance import mean_variance_optimization
from optimizer.models.risk_parity import *
# This decorator ensures that only logged-in users can access this page
@login_required
def welcome_view(request):
    """
    Renders the main welcome page with navigation buttons.
    """
    return render(request, 'optimizer/welcome.html')

@login_required
def know_the_models_view(request):
    """
    Renders the page explaining the different portfolio models.
    """
    return render(request, 'optimizer/know_the_models.html')

@login_required
def portfolio_optimizer_view(request):
    """
    Handles file upload, model selection, and displays optimization results.
    """
    if request.method == 'POST':
        # Use the new form that includes the model choice dropdown
        form = PortfolioOptimizerForm(request.POST, request.FILES)
        if form.is_valid():
            # Get data from the validated form
            selected_model = form.cleaned_data['model_choice']
            returns_file = form.cleaned_data['returns_file']
            
            try:
                # Read the uploaded CSV file with pandas
                df = pd.read_csv(returns_file)
                df = df.select_dtypes(include=['number'])  # Ensure only numeric columns are processed
                df = df.dropna()  # Drop rows with NaN values
                # --- ADD YOUR MODEL LOGIC HERE ---
                # Based on the user's choice, run the appropriate model.
                # The result should be stored in the 'results_html' variable.
                
                if selected_model == 'mean_variance':
                    # Call your mean-variance function with the dataframe 'df'
                    # results_html = your_mean_variance_function(df)
                    results_html = mean_variance_optimization(df)
                elif selected_model == 'risk_parity':
                    results_html = risk_parity(df)
                    return render(request, 'optimizer/portfolio_optimizer.html', {
                        'form': form,
                        'optimization_results': results_html
                    })     
                    

                    #results_html = df.head().to_html(classes='table-auto w-full text-left whitespace-no-wrap') + "<p class='mt-2 font-bold'>Processed with Risk Parity.</p>"

                else:
                    results_html = "<p>Selected model not recognized.</p>"

                return render(request, 'optimizer/portfolio_optimizer.html', {
                    'form': form,
                    'optimization_results': results_html
                })

            except Exception as e:
                # Handle potential errors in file reading or processing
                error_message = f"Error processing file: {e}"
                return render(request, 'optimizer/portfolio_optimizer.html', {
                    'form': form,
                    'error_message': error_message
                })
    else:
        # For a GET request, just display a blank form
        form = PortfolioOptimizerForm()

    return render(request, 'optimizer/portfolio_optimizer.html', {'form': form})

@login_required
def about_us_view(request):
    """
    Renders the 'About Us' page.
    """
    return render(request, 'optimizer/about_us.html')

@login_required
def contact_us_view(request):
    """
    Renders the 'Contact Us' page.
    """
    return render(request, 'optimizer/contact_us.html')