 InvestoQuest – Portfolio Optimization \& Management



This project integrates Modern Portfolio Theory (MPT), Hierarchical Risk Parity (HRP), and Dynamic Weight Allocation for optimal portfolio construction.  

It includes Python-based analytics, transaction cost modeling, and a Django web app for real-time optimization.



Features

- Efficient Frontier Visualization (MPT)

- HRP \& HERC Portfolio Optimization

- Dynamic Rebalancing with Transaction Costs

- Django-based Interactive Dashboard



Methodology

- MPT optimization using covariance matrix

- HRP \& HERC clustering workflow

- Correlation-based dynamic rebalancing




Tech Stack

- Python (NumPy, Pandas, Matplotlib, SciPy, Riskfolio-Lib)

- Django Framework for Web Deployment

- yFinance for live stock data retrieval




Results

- Efficient Frontier visualization with maximum Sharpe ratio

- Hierarchical clustering dendrograms

- Adaptive rebalancing with lower transaction costs



Report

See full documentation in [`FAC_Report.pdf`](FAC_Report.pdf)




Author: Nisarg Parashar  

Institute: IIT Kanpur (Finance & Analytics Club)
InvestoQuest – Portfolio Optimization & Management


To run this Django-based portfolio optimization project on your own system, first clone the repository using `git clone https://github.com/nisarg3176/InvestoQuest-Portfolio-Optimization.git` and move into the folder with `cd InvestoQuest-Portfolio-Optimization`. Next, create a virtual environment by running `python -m venv venv` and activate it using `source venv/Scripts/activate` on Windows or `source venv/bin/activate` on Mac/Linux. Then install all required dependencies with `pip install -r requirements.txt`. After installation, set up the database using `python manage.py migrate`. (Optionally, create an admin account with `python manage.py createsuperuser` to access the Django admin panel.) Finally, start the server using `python manage.py runserver` and open your browser to `http://127.0.0.1:8000/` to view the project. You can access the admin dashboard at `http://127.0.0.1:8000/admin/` using the superuser credentials. If you face any missing module errors (like `No module named 'pandas'`), simply install them manually using `pip install pandas numpy django matplotlib`. Once the server is running, you can explore all portfolio optimization features and analytics directly from your browser.



