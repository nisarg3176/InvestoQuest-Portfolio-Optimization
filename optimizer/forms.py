# optimizer/forms.py
from django import forms

class PortfolioOptimizerForm(forms.Form):
    # Your model choices are perfect, no changes needed here.
    MODEL_CHOICES = [
        ('mean_variance', 'Mean-Variance Optimization'),
        ('black_litterman', 'Black-Litterman Model'),
        ('risk_parity', 'Risk Parity')
    ]

    # --- CHANGE 1: SIMPLIFIED THE FILE FIELD ---
    # We removed the custom widget because the template now handles the UI
    # for file uploads with a custom "drag-and-drop" zone.
    returns_file = forms.FileField(
        label="Select a returns CSV file",
        help_text="File must be in CSV format.",
    )

    # --- CHANGE 2: UPDATED THE DROPDOWN STYLING ---
    # Replaced the old light-theme classes with the new dark-theme ones.
    model_choice = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label="Select Optimization Model",
        widget=forms.Select(attrs={
            'class': 'block w-full rounded-lg border border-white/20 bg-white/5 px-4 py-3 text-white placeholder-gray-400 transition duration-300 ease-in-out focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-400/50'
        })
    )

    # Your validation function is great, no changes needed here.
    def clean_returns_file(self):
        """
        Validates that the uploaded file is a CSV.
        """
        file = self.cleaned_data.get('returns_file')
        if file and not file.name.endswith('.csv'):
            raise forms.ValidationError("Only CSV files are accepted.")
        return file